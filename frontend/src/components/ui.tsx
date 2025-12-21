import { type ButtonHTMLAttributes, type InputHTMLAttributes, type PropsWithChildren } from 'react'

export function Button({ className = '', ...props }: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      className={`inline-flex items-center justify-center rounded-md border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-900 shadow-sm transition hover:-translate-y-[1px] hover:shadow focus:outline-none focus:ring-2 focus:ring-slate-300 ${className}`}
      {...props}
    />
  )
}

export function PrimaryButton({ className = '', ...props }: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      className={`inline-flex items-center justify-center rounded-md bg-slate-900 px-3 py-2 text-sm font-semibold text-white shadow transition hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-slate-300 disabled:cursor-not-allowed disabled:bg-slate-300 ${className}`}
      {...props}
    />
  )
}

export function Input({ className = '', ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={`w-full rounded-md border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm placeholder:text-slate-400 focus:border-slate-400 focus:outline-none focus:ring-2 focus:ring-slate-300 ${className}`}
      {...props}
    />
  )
}

export function Panel({ className = '', children }: PropsWithChildren<{ className?: string }>) {
  return <div className={`rounded-lg border border-slate-200 bg-white shadow-sm ${className}`}>{children}</div>
}

export function Badge({ className = '', children }: PropsWithChildren<{ className?: string }>) {
  return (
    <span className={`inline-flex items-center rounded-full border border-slate-200 bg-slate-50 px-2 py-1 text-xs font-medium text-slate-600 ${className}`}>
      {children}
    </span>
  )
}
