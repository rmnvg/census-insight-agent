export function LogoMark({ className = "size-7" }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden>
      <rect width="32" height="32" rx="8" className="fill-teal-700" />
      <rect x="7" y="16" width="4" height="9" rx="1.5" fill="#fff" />
      <rect x="14" y="11" width="4" height="14" rx="1.5" fill="#fff" />
      <rect x="21" y="7" width="4" height="18" rx="1.5" fill="#99f6e4" />
    </svg>
  );
}
