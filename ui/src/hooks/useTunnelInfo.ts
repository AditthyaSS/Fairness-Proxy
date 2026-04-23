// src/hooks/useTunnelInfo.ts
import { useEffect } from 'react'
import { usePipelineStore } from '../store/usePipelineStore'

/**
 * Auto-detect ngrok public URL on load.
 * Only switches to ngrok URL if the frontend is NOT running on localhost
 * (i.e., deployed somewhere else and needs to find the API).
 * When running locally, keep using localhost:8000 directly.
 */
export function useTunnelInfo() {
  useEffect(() => {
    // If we're on localhost, don't switch — direct connection is faster
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
      return
    }

    // For non-local deployments, try to discover the tunnel URL
    fetch('http://localhost:8000/v1/tunnel/info')
      .then(r => r.json())
      .then(data => {
        if (data.tunnel_active && data.public_url) {
          usePipelineStore.getState().setBaseUrl(data.public_url)
          console.log('🚀 Auto-configured to live URL:', data.public_url)
        }
      })
      .catch(() => {
        // Stay on current baseUrl if tunnel info unavailable
      })
  }, [])
}
