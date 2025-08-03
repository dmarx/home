// quartz/components/scripts/graph.inline.ts - Debug Version
import type { ContentDetails } from "../../plugins/emitters/contentIndex"
import { Cosmograph } from "@cosmograph/cosmograph"
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
  return Math.max(4, 2 + Math.sqrt(numLinks))
}

async function renderGraph(container: HTMLElement, fullSlug: FullSlug): Promise<() => void> {
  console.log('🎯 Starting graph render for:', fullSlug)
  
  const slug = simplifySlug(fullSlug)
  const visited = getVisited()
  removeAllChildren(container)

  // Ensure container has dimensions
  if (!container.style.height) {
    container.style.height = '400px'
    console.log('📏 Set container height to 400px')
  }
  if (!container.style.width) {
    container.style.width = '100%'
    console.log('📏 Set container width to 100%')
  }

  console.log('📦 Container dimensions:', {
    width: container.offsetWidth,
    height: container.offsetHeight,
    style: container.style.cssText
  })

  const config = JSON.parse(container.dataset["cfg"]!) as CosmographConfig
  console.log('⚙️ Graph config:', config)

  // Check if fetchData is available
  if (typeof fetchData === 'undefined') {
    console.error('❌ fetchData is not available globally')
    container.innerHTML = '<div style="padding: 20px; color: red;">Error: Graph data not available</div>'
    return () => {}
  }

  try {
    // Load data
    const rawData = await fetchData
    console.log('📊 Raw data loaded:', Object.keys(rawData).length, 'entries')
    
    const data: Map<SimpleSlug, ContentDetails> = new Map(
      Object.entries<ContentDetails>(rawData).map(([k, v]) => [
        simplifySlug(k as FullSlug),
        v,
      ]),
    )

    const links: LinkData[] = []
    const tags: SimpleSlug[] = []
    const validLinks = new Set(data.keys())

    console.log('🔗 Building links and collecting tags...')

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

    console.log('🔗 Total links found:', links.length)

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

    console.log('🏘️ Neighbourhood size:', neighbourhood.size)

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

    console.log('👥 Nodes to render:', nodes.length)
    console.log('🔗 Links to render:', filteredLinks.length)

    if (nodes.length === 0) {
      console.warn('⚠️ No nodes to render!')
      container.innerHTML = '<div style="padding: 20px; color: orange;">No nodes found for this page</div>'
      return () => {}
    }

    // Create canvas
    const canvas = document.createElement('canvas')
    canvas.style.width = '100%'
    canvas.style.height = '100%'
    canvas.style.background = 'transparent'
    canvas.style.border = '1px solid red' // Debug border
    container.appendChild(canvas)

    console.log('🎨 Canvas created and added to container')
    console.log('📐 Canvas dimensions after creation:', {
      width: canvas.width,
      height: canvas.height,
      offsetWidth: canvas.offsetWidth,
      offsetHeight: canvas.offsetHeight
    })

    // Wait a bit for canvas to be properly sized
    await new Promise(resolve => setTimeout(resolve, 100))

    console.log('📐 Canvas dimensions after timeout:', {
      width: canvas.width,
      height: canvas.height,
      offsetWidth: canvas.offsetWidth,
      offsetHeight: canvas.offsetHeight
    })

    // Initialize cosmograph with minimal config first
    console.log('🚀 Initializing Cosmograph...')
    
    try {
      const cosmograph = new Cosmograph(canvas, {
        simulation: {
          repulsion: config.repulsion || 0.5,
          linkSpring: config.linkSpring || 1.0,
          linkDistance: config.linkDistance || 10,
          friction: config.friction || 0.85,
          gravity: config.gravity || 0.1,
        },
        renderLinks: true,
        nodeColor: (node: NodeData) => {
          const color = node.color || config.nodeColor || '#8b5cf6'
          console.log('🎨 Node color for', node.id, ':', color)
          return color
        },
        nodeSize: (node: NodeData) => node.size || config.nodeSize || 4,
        linkColor: config.linkColor || '#64748b',
        linkWidth: config.linkWidth || 1,
        backgroundColor: config.backgroundColor || 'transparent',
        showDynamicLabels: config.showDynamicLabels ?? true,
        events: {
          onClick: (node?: NodeData) => {
            console.log('🖱️ Node clicked:', node?.id)
            if (node) {
              const target = resolveRelative(fullSlug, node.id)
              window.spaNavigate(new URL(target, window.location.toString()))
            }
          }
        }
      })

      console.log('✅ Cosmograph initialized successfully')

      // Set data
      console.log('📊 Setting graph data...')
      cosmograph.setData(nodes, filteredLinks)
      console.log('✅ Data set successfully')

      // Try to fit view
      setTimeout(() => {
        try {
          console.log('🔍 Attempting to fit view...')
          cosmograph.fitView()
          console.log('✅ View fitted successfully')
        } catch (e) {
          console.error('❌ Error fitting view:', e)
        }
      }, 500)

      // Handle resize
      const resizeObserver = new ResizeObserver(() => {
        console.log('📏 Container resized, fitting view...')
        cosmograph.fitView()
      })
      resizeObserver.observe(container)

      console.log('✅ Graph render completed successfully')

      // Cleanup function
      return () => {
        console.log('🧹 Cleaning up graph')
        resizeObserver.disconnect()
        cosmograph.destroy()
        removeAllChildren(container)
      }

    } catch (error) {
      console.error('❌ Error initializing Cosmograph:', error)
      container.innerHTML = `<div style="padding: 20px; color: red;">Error initializing graph: ${error.message}</div>`
      return () => {}
    }

  } catch (error) {
    console.error('❌ Error loading graph data:', error)
    container.innerHTML = `<div style="padding: 20px; color: red;">Error loading graph data: ${error.message}</div>`
    return () => {}
  }
}

let localGraphCleanups: (() => void)[] = []
let globalGraphCleanups: (() => void)[] = []

function cleanupLocalGraphs() {
  console.log('🧹 Cleaning up local graphs')
  for (const cleanup of localGraphCleanups) {
    cleanup()
  }
  localGraphCleanups = []
}

function cleanupGlobalGraphs() {
  console.log('🧹 Cleaning up global graphs')
  for (const cleanup of globalGraphCleanups) {
    cleanup()
  }
  globalGraphCleanups = []
}

document.addEventListener("nav", async (e: CustomEventMap["nav"]) => {
  console.log('🧭 Navigation event triggered for:', e.detail.url)
  
  const slug = e.detail.url
  addToVisited(simplifySlug(slug))

  async function renderLocalGraph() {
    console.log('🏠 Rendering local graphs...')
    cleanupLocalGraphs()
    const localGraphContainers = document.getElementsByClassName("graph-container")
    console.log('📦 Found', localGraphContainers.length, 'local graph containers')
    
    for (const container of localGraphContainers) {
      try {
        localGraphCleanups.push(await renderGraph(container as HTMLElement, slug))
      } catch (error) {
        console.error('❌ Failed to render local graph:', error)
      }
    }
  }

  await renderLocalGraph()
  
  const handleThemeChange = () => {
    console.log('🎨 Theme changed, re-rendering graphs...')
    void renderLocalGraph()
  }

  document.addEventListener("themechange", handleThemeChange)
  window.addCleanup(() => {
    document.removeEventListener("themechange", handleThemeChange)
  })

  const containers = [...document.getElementsByClassName("global-graph-outer")] as HTMLElement[]
  
  async function renderGlobalGraph() {
    console.log('🌍 Rendering global graphs...')
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
        try {
          globalGraphCleanups.push(await renderGraph(graphContainer, slug))
        } catch (error) {
          console.error('❌ Failed to render global graph:', error)
        }
      }
    }
  }

  function hideGlobalGraph() {
    console.log('🙈 Hiding global graphs...')
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
