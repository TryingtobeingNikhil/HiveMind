import { useState, useEffect } from 'react'
import { checkHealth } from '../utils/api'

/**
 * Polls the backend health endpoint on mount and every 30 seconds.
 * Returns { isOnline: boolean }.
 *
 * The interval is cleared on unmount — no leaks.
 */
export function useHealthCheck() {
  const [isOnline, setIsOnline] = useState(false)

  useEffect(() => {
    let cancelled = false

    async function ping() {
      const ok = await checkHealth()
      if (!cancelled) setIsOnline(ok)
    }

    ping()
    const intervalId = setInterval(ping, 30_000)

    return () => {
      cancelled = true
      clearInterval(intervalId)
    }
  }, [])

  return { isOnline }
}
