/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,jsx,ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        'bg-base':    'var(--bg-base)',
        'bg-surface': 'var(--bg-surface)',
        'bg-raised':  'var(--bg-raised)',
        'bg-hover':   'var(--bg-hover)',
        'text-primary':   'var(--text-primary)',
        'text-secondary': 'var(--text-secondary)',
        'text-muted':     'var(--text-muted)',
        'accent':     'var(--accent)',
        'accent-dim': 'var(--accent-dim)',
        'border-subtle': 'var(--border-subtle)',
        'stage-planning':    'var(--stage-planning)',
        'stage-researching': 'var(--stage-researching)',
        'stage-critiquing':  'var(--stage-critiquing)',
        'stage-reporting':   'var(--stage-reporting)',
        'stage-evaluating':  'var(--stage-evaluating)',
        'stage-completed':   'var(--stage-completed)',
        'stage-failed':      'var(--stage-failed)',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      keyframes: {
        'pulse-ring': {
          '0%':   { boxShadow: '0 0 0 0 currentColor', opacity: '0.6' },
          '100%': { boxShadow: '0 0 0 8px currentColor', opacity: '0' },
        },
        'fade-in-up': {
          'from': { opacity: '0', transform: 'translateY(8px)' },
          'to':   { opacity: '1', transform: 'translateY(0)' },
        },
        'fade-in': {
          'from': { opacity: '0' },
          'to':   { opacity: '1' },
        },
      },
      animation: {
        'pulse-ring': 'pulse-ring 1.5s ease-out infinite',
        'fade-in-up': 'fade-in-up 300ms ease-out forwards',
        'fade-in':    'fade-in 200ms ease-out forwards',
      },
    },
  },
  plugins: [],
}
