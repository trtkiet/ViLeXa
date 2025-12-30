import { Navigate, NavLink, Route, Routes } from 'react-router-dom';
import { ChatPage } from './pages/ChatPage';
import { LookupPage } from './pages/LookupPage';

export default function App() {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-2 text-lg font-semibold">
            <span className="bg-slate-900 text-white px-2 py-1 rounded">VA</span>
            <span>Law Assistant</span>
          </div>
          <nav className="flex gap-4">
            <NavLink to="/" className={({isActive}) => isActive ? "font-bold underline" : ""}>Chat</NavLink>
            <NavLink to="/lookup" className={({isActive}) => isActive ? "font-bold underline" : ""}>Lookup</NavLink>
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-8">
        <Routes>
          <Route path="/" element={<ChatPage />} />
          <Route path="/lookup" element={<LookupPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}