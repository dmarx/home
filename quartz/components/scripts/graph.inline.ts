// quartz/components/scripts/graph.inline.ts
import type { ContentDetails } from "../../plugins/emitters/contentIndex"
import { Graph } from "@cosmograph/cosmograph"
import { registerEscapeHandler, removeAllChildren } from "./util"
import { FullSlug, SimpleSlug, getFullSlug, resolveRelative, simplifySlug } from "../../util/path"
import { CosmographConfig } from "../Graph"

type NodeData = {
  id: SimpleSlug
  text: string
  tags: string[]
  color?: string
  size?: number
  visited?: boolean
  isCurrent?: boolean
  isTag?: boolean
}

type LinkData = {
  source: SimpleSlug
  target: SimpleSlug
}

const localStorageKey = "graph-visited"

function getVisited(): Set<SimpleSlug> {
  return new Set(JSON.parse(localStorage.getItem(localStorageKey) ?? "[]"))
}

function addToVisited(slug: SimpleSlug) {
  const visited = getVisited()
  visited.add(slug)
  localStorage.setItem(localStorageKey, JSON.stringify([...visited]))
}

function getComputedCSSColor(variable: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(variable).trim()
}

function getNodeColor(node: NodeData): string {
  if (node.isCurrent) {
    return getComputedCSSColor('--secondary') || '#8b5cf6'
  } else if (node.visited || node.isTag) {
    return getComputedCSSColor('--tertiary') || '#06b6d4'
  } else {
    return getComputedCSSColor('--gray') || '#6b7280'
  }
}

function getNodeSize(node: NodeData, links: LinkData[]): number {
  const numLinks = links.filter(
    (l) => l.source === node.id || l.target === node.id,
  ).length
  return Math.max(2, 2 + Math.sqrt(numLinks))
}

async function renderGraph(container: HTMLElement, fullSlug: FullSlug): Promise<() => void> {
  const slug = simplifySlug(fullSlug)
  const visited = getVisited()
  removeAllChildren(container)

  const config = JSON.parse(container.dataset["cfg"]!) as CosmographConfig

  // Load data
  const data: Map<SimpleSlug, ContentDetails> = new Map(
    Object.entries<ContentDetails>(await fetchData).map(([k, v]) => [
      simplifySlug(k as FullSlug),
      v,
    ]),
  )

  const links: LinkData[] = []
  const tags: SimpleSlug[] = []
  const validLinks = new Set(data.keys())

  // Build links and collect tags
  for (const [source, details] of data.entries()) {
    const outgoing = details.links ?? []

    for (const dest of outgoing) {
      if (validLinks.has(dest)) {
        links.push({ source: source, target: dest })
      }
    }

    if (config.showTags) {
      const localTags = details.tags
        .filter((tag) => !config.removeTags.includes(tag))
        .map((tag) => simplifySlug(("tags/" + tag) as FullSlug))

      tags.push(...localTags.filter((tag) => !tags.includes(tag)))

      for (const tag of localTags) {
        links.push({ source: source, target: tag })
      }
    }
  }

  // Build neighborhood based on depth
  const neighbourhood = new Set<SimpleSlug>()
  const wl: (SimpleSlug | "__SENTINEL")[] = [slug, "__SENTINEL"]
  let depth = config.depth

  if (depth >= 0) {
    while (depth >= 0 && wl.length > 0) {
      const cur = wl.shift()!
      if (cur === "__SENTINEL") {
        depth--
        wl.push("__SENTINEL")
      } else {
        neighbourhood.add(cur)
        const outgoing = links.filter((l) => l.source === cur)
        const incoming = links.filter((l) => l.target === cur)
        wl.push(...outgoing.map((l) => l.target), ...incoming.map((l) => l.source))
      }
    }
  } else {
    validLinks.forEach((id) => neighbourhood.add(id))
    if (config.showTags) tags.forEach((tag) => neighbourhood.add(tag))
  }

  // Create nodes
  const nodes: NodeData[] = [...neighbourhood].map((url) => {
    const isTag = url.startsWith("tags/")
    const text = isTag ? "#" + url.substring(5) : (data.get(url)?.title ?? url)
    const nodeData: NodeData = {
      id: url,
      text,
      tags: data.get(url)?.tags ?? [],
      visited: visited.has(url),
      isCurrent: url === slug,
      isTag,
    }
    
    nodeData.color = getNodeColor(nodeData)
    nodeData.size = getNodeSize(nodeData, links)
    
    return nodeData
  })

  // Filter links to only include nodes in neighbourhood
  const filteredLinks = links.filter(
    (l) => neighbourhood.has(l.source) && neighbourhood.has(l.target)
  )

  // Create canvas
  const canvas = document.createElement('canvas')
  container.appendChild(canvas)

  // Initialize cosmograph
  const cosmograph = new Graph(canvas, {
    // Simulation settings
    simulation: {
      repulsion: config.repulsion,
      repulsionTheta: config.repulsionTheta,
      repulsionQuadtreeLevels: config.repulsionQuadtreeLevels,
      linkSpring: config.linkSpring,
      linkDistance: config.linkDistance,
      friction: config.friction,
      gravity: config.gravity,
      center: config.gravity > 0,
    },
    
    // Rendering settings
    renderLinks: true,
    renderHoveredNodeRing: true,
    nodeColor: (node: NodeData) => node.color || config.nodeColor,
    nodeSize: (node: NodeData) => node.size || config.nodeSize,
    linkColor: config.linkColor,
    linkWidth: config.linkWidth,
    linkArrows: config.linkArrows,
    
    // Background
    backgroundColor: config.backgroundColor,
    
    // Labels
    showLabelsFor: config.showLabels ? nodes : [],
    showDynamicLabels: config.showDynamicLabels,
    nodeLabelAccessor: (node: NodeData) => node.text,
    nodeLabelColor: config.labelColor,
    hoveredNodeLabelColor: config.hoveredNodeLabelColor,
    
    // Interaction
    disableSimulation: config.disableSimulation,
    
    // Events
    onClick: (node?: NodeData) => {
      if (node) {
        const target = resolveRelative(fullSlug, node.id)
        window.spaNavigate(new URL(target, window.location.toString()))
      }
    },
    
    onNodeMouseOver: (node?: NodeData) => {
      if (config.focusOnHover && node) {
        // Get connected nodes
        const connectedNodeIds = new Set<SimpleSlug>()
        filteredLinks.forEach(link => {
          if (link.source === node.id) {
            connectedNodeIds.add(link.target)
          }
          if (link.target === node.id) {
            connectedNodeIds.add(link.source)
          }
        })
        connectedNodeIds.add(node.id)
        
        // Highlight connected nodes and dim others
        cosmograph.setConfig({
          nodeColor: (n: NodeData) => {
            if (connectedNodeIds.has(n.id)) {
              return n.color || config.nodeColor
            } else {
              // Return dimmed version of the color
              const originalColor = n.color || config.nodeColor
              return originalColor + '40' // Add alpha for dimming
            }
          }
        })
      }
    },
    
    onNodeMouseOut: () => {
      if (config.focusOnHover) {
        // Reset colors
        cosmograph.setConfig({
          nodeColor: (node: NodeData) => node.color || config.nodeColor
        })
      }
    }
  })

  // Set data
  cosmograph.setData(nodes, filteredLinks)

  // Fit to container
  cosmograph.fitView()

  // Handle resize
  const resizeObserver = new ResizeObserver(() => {
    cosmograph.fitView()
  })
  resizeObserver.observe(container)

  // Cleanup function
  return () => {
    resizeObserver.disconnect()
    cosmograph.destroy()
    removeAllChildren(container)
  }
}

let localGraphCleanups: (() => void)[] = []
let globalGraphCleanups: (() => void)[] = []

function cleanupLocalGraphs() {
  for (const cleanup of localGraphCleanups) {
    cleanup()
  }
  localGraphCleanups = []
}

function cleanupGlobalGraphs() {
  for (const cleanup of globalGraphCleanups) {
    cleanup()
  }
  globalGraphCleanups = []
}

document.addEventListener("nav", async (e: CustomEventMap["nav"]) => {
  const slug = e.detail.url
  addToVisited(simplifySlug(slug))

  async function renderLocalGraph() {
    cleanupLocalGraphs()
    const localGraphContainers = document.getElementsByClassName("graph-container")
    for (const container of localGraphContainers) {
      localGraphCleanups.push(await renderGraph(container as HTMLElement, slug))
    }
  }

  await renderLocalGraph()
  
  const handleThemeChange = () => {
    void renderLocalGraph()
  }

  document.addEventListener("themechange", handleThemeChange)
  window.addCleanup(() => {
    document.removeEventListener("themechange", handleThemeChange)
  })

  const containers = [...document.getElementsByClassName("global-graph-outer")] as HTMLElement[]
  
  async function renderGlobalGraph() {
    cleanupGlobalGraphs()
    const slug = getFullSlug(window)
    for (const container of containers) {
      container.classList.add("active")
      const sidebar = container.closest(".sidebar") as HTMLElement
      if (sidebar) {
        sidebar.style.zIndex = "1"
      }

      const graphContainer = container.querySelector(".global-graph-container") as HTMLElement
      registerEscapeHandler(container, hideGlobalGraph)
      if (graphContainer) {
        globalGraphCleanups.push(await renderGraph(graphContainer, slug))
      }
    }
  }

  function hideGlobalGraph() {
    cleanupGlobalGraphs()
    for (const container of containers) {
      container.classList.remove("active")
      const sidebar = container.closest(".sidebar") as HTMLElement
      if (sidebar) {
        sidebar.style.zIndex = ""
      }
    }
  }

  async function shortcutHandler(e: HTMLElementEventMap["keydown"]) {
    if (e.key === "g" && (e.ctrlKey || e.metaKey) && !e.shiftKey) {
      e.preventDefault()
      const anyGlobalGraphOpen = containers.some((container) =>
        container.classList.contains("active"),
      )
      anyGlobalGraphOpen ? hideGlobalGraph() : renderGlobalGraph()
    }
  }

  const containerIcons = document.getElementsByClassName("global-graph-icon")
  Array.from(containerIcons).forEach((icon) => {
    icon.addEventListener("click", renderGlobalGraph)
    window.addCleanup(() => icon.removeEventListener("click", renderGlobalGraph))
  })

  document.addEventListener("keydown", shortcutHandler)
  window.addCleanup(() => {
    document.removeEventListener("keydown", shortcutHandler)
    cleanupLocalGraphs()
    cleanupGlobalGraphs()
  })
})
