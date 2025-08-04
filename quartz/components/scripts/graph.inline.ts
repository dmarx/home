// quartz/components/scripts/graph.inline.ts - Fixed Version
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

    // Create graph container div
    const graphDiv = document.createElement('div')
    graphDiv.style.width = '100%'
    graphDiv.style.height = '100%'
    graphDiv.style.position = 'relative'
    container.appendChild(graphDiv)

    console.log('🎨 Graph div created and added to container')

    // Wait for layout
    await new Promise(resolve => setTimeout(resolve, 100))

    console.log('🚀 Initializing Cosmograph with DIV...')
    
    try {
      // Create cosmograph with the DIV (not canvas)
      const cosmograph = new Cosmograph(graphDiv, {
        // Simulation settings - use simpler values for testing
        simulation: {
          repulsion: 1.0,  // Increased for better separation
          linkSpring: 0.5,
          linkDistance: 50, // Increased for visibility
          friction: 0.8,
          gravity: 0.2,    // Increased to center nodes
        },
        
        // Essential rendering settings
        renderLinks: true,
        nodeColor: (node: NodeData) => {
          const color = node.color || '#8b5cf6'
          console.log('🎨 Setting node color for', node.id, ':', color)
          return color
        },
        nodeSize: (node: NodeData) => {
          const size = (node.size || 8) * 2 // Make nodes bigger for visibility
          console.log('📏 Setting node size for', node.id, ':', size)
          return size
        },
        linkColor: '#94a3b8',
        linkWidth: 2, // Make links thicker for visibility
        
        // Background and viewport
        backgroundColor: 'rgba(0,0,0,0)', // Transparent
        pixelRatio: window.devicePixelRatio || 1,
        
        // Labels
        showDynamicLabels: true,
        
        // Initial view settings - CRITICAL for visibility
        fitViewOnInit: true,
        fitViewDelay: 500,
        fitViewPadding: 0.1,
        
        // Events
        events: {
          onClick: (node?: NodeData, event?: any) => {
            console.log('🖱️ CLICK EVENT DETECTED!', { node, event, hasNodeId: !!node?.id })
            
            if (node && node.id) {
              console.log('🧭 Navigating to:', node.id)
              try {
                const target = resolveRelative(fullSlug, node.id)
                console.log('🔗 Resolved target URL:', target)
                
                // Check if spaNavigate is available
                if (typeof window.spaNavigate === 'function') {
                  window.spaNavigate(new URL(target, window.location.toString()))
                  console.log('✅ Navigation initiated via SPA')
                } else {
                  // Fallback to regular navigation
                  console.log('⚠️ SPA navigation not available, using fallback')
                  window.location.href = target
                }
              } catch (error) {
                console.error('❌ Navigation error:', error)
                // Fallback navigation
                window.location.href = node.id
              }
            } else {
              console.log('⚠️ No node or node.id in click event', node)
            }
          },
          
          onNodeMouseOver: (node?: NodeData) => {
            if (config.focusOnHover && node) {
              console.log('🏠 Hovering over node:', node.id)
              // TODO: Implement focus highlight
            }
          },
          
          onNodeMouseOut: () => {
            if (config.focusOnHover) {
              console.log('👋 Mouse left node')
              // TODO: Reset focus highlight
            }
          }
        }
      })

      console.log('✅ Cosmograph initialized successfully')
      console.log('🔧 Cosmograph config applied, onClick should be registered')

      // Set data
      console.log('📊 Setting graph data...')
      console.log('Sample nodes:', nodes.slice(0, 3).map(n => ({ id: n.id, text: n.text, color: n.color })))
      console.log('Sample links:', filteredLinks.slice(0, 3))
      
      cosmograph.setData(nodes, filteredLinks)
      console.log('✅ Data set successfully')
      
      // Add debugging for click events
      console.log('🖱️ Click handler should be ready for nodes:', nodes.map(n => n.id).slice(0, 5))

      // Multiple attempts to fit view for visibility
      const fitViewAttempts = [100, 500, 1000, 2000]
      fitViewAttempts.forEach(delay => {
        setTimeout(() => {
          try {
            console.log(`🔍 Attempting to fit view (${delay}ms)...`)
            cosmograph.fitView()
            console.log('✅ View fitted successfully')
          } catch (e) {
            console.error('❌ Error fitting view:', e)
          }
        }, delay)
      })

      // Handle resize
      const resizeObserver = new ResizeObserver(() => {
        console.log('📏 Container resized, fitting view...')
        setTimeout(() => cosmograph.fitView(), 100)
      })
      resizeObserver.observe(container)

      console.log('✅ Graph render completed successfully')

      // Cleanup function
      return () => {
        console.log('🧹 Cleaning up graph')
        resizeObserver.disconnect()
        try {
          // Check if cosmograph has destroy method before calling it
          if (cosmograph && typeof cosmograph.destroy === 'function') {
            cosmograph.destroy()
          } else if (cosmograph && typeof cosmograph.clear === 'function') {
            cosmograph.clear()
          }
        } catch (error) {
          console.warn('⚠️ Error during cosmograph cleanup:', error)
        }
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
  console.log('📦 Found global graph containers:', containers.length)
  
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
      
      // Use the existing registerEscapeHandler - it handles both escape key and click-outside
      // The container (.global-graph-outer) is the backdrop that should be clickable
      registerEscapeHandler(container, hideGlobalGraph)
      console.log('⌨️ Escape handler registered for global graph')
      
      if (graphContainer) {
        try {
          const cleanup = await renderGraph(graphContainer, slug)
          globalGraphCleanups.push(cleanup)
        } catch (error) {
          console.error('❌ Failed to render global graph:', error)
        }
      }
    }
  }

  function hideGlobalGraph() {
    console.log('🙈 Hiding global graphs...')
    
    // First cleanup the graph instances
    cleanupGlobalGraphs()
    
    // Get fresh container references (in case DOM changed)
    const currentContainers = [...document.getElementsByClassName("global-graph-outer")] as HTMLElement[]
    console.log('📦 Current containers for hiding:', currentContainers.length)
    
    // Then hide the overlay containers
    for (let i = 0; i < currentContainers.length; i++) {
      const container = currentContainers[i]
      console.log(`🙈 Processing container ${i}:`, {
        hasActiveClass: container.classList.contains('active'),
        display: getComputedStyle(container).display,
        classList: Array.from(container.classList)
      })
      
      container.classList.remove("active")
      
      console.log(`🙈 After removing active class:`, {
        hasActiveClass: container.classList.contains('active'),
        display: getComputedStyle(container).display,
        classList: Array.from(container.classList)
      })
      
      const sidebar = container.closest(".sidebar") as HTMLElement
      if (sidebar) {
        sidebar.style.zIndex = ""
      }
    }
    
    console.log('✅ Global graph hidden successfully')
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
