/**
 * Sidebar component with session list and user menu
 */

import { useState, useRef, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import type { ChatSession } from '../../types/session';

interface SidebarProps {
  sessions: ChatSession[];
  activeSessionId: string | null;
  onSelectSession: (sessionId: string | null) => void;
  onNewSession: () => void;
  onDeleteSession: (sessionId: string) => void;
  onRenameSession: (sessionId: string, title: string) => void;
  isLoading?: boolean;
}

export function Sidebar({
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  onRenameSession,
  isLoading,
}: SidebarProps) {
  const { user, logout } = useAuth();
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const userMenuRef = useRef<HTMLDivElement>(null);

  // Close menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpenId(null);
      }
      if (userMenuRef.current && !userMenuRef.current.contains(e.target as Node)) {
        setUserMenuOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Group sessions by date
  const groupedSessions = groupSessionsByDate(sessions);

  const handleRename = (sessionId: string) => {
    if (editTitle.trim()) {
      onRenameSession(sessionId, editTitle.trim());
    }
    setEditingId(null);
    setEditTitle('');
  };

  const startEditing = (session: ChatSession) => {
    setEditingId(session.id);
    setEditTitle(session.title);
    setMenuOpenId(null);
  };

  return (
    <div className="w-64 h-full bg-gray-50 border-r border-gray-200 flex flex-col">
      {/* New Chat Button */}
      <div className="p-3">
        <button
          onClick={onNewSession}
          className="w-full flex items-center gap-2 px-3 py-2.5 text-sm font-medium text-gray-700 bg-white border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          New Chat
        </button>
      </div>

      {/* Sessions List */}
      <div className="flex-1 overflow-y-auto px-2 pb-2">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <div className="w-5 h-5 border-2 border-gray-300 border-t-gray-600 rounded-full animate-spin" />
          </div>
        ) : sessions.length === 0 ? (
          <p className="px-3 py-4 text-sm text-gray-500 text-center">
            No conversations yet
          </p>
        ) : (
          Object.entries(groupedSessions).map(([group, groupSessions]) => (
            <div key={group} className="mb-4">
              <h3 className="px-3 py-1 text-xs font-medium text-gray-500 uppercase tracking-wider">
                {group}
              </h3>
              {groupSessions.map(session => (
                <div key={session.id} className="relative" ref={menuOpenId === session.id ? menuRef : null}>
                  {editingId === session.id ? (
                    <form
                      onSubmit={e => {
                        e.preventDefault();
                        handleRename(session.id);
                      }}
                      className="px-1 py-0.5"
                    >
                      <input
                        type="text"
                        value={editTitle}
                        onChange={e => setEditTitle(e.target.value)}
                        onBlur={() => handleRename(session.id)}
                        autoFocus
                        className="w-full px-2 py-1.5 text-sm border border-blue-500 rounded focus:outline-none"
                      />
                    </form>
                  ) : (
                    <button
                      onClick={() => onSelectSession(session.id)}
                      className={`w-full flex items-center justify-between gap-2 px-3 py-2 text-sm rounded-lg transition-colors group ${
                        activeSessionId === session.id
                          ? 'bg-gray-200 text-gray-900'
                          : 'text-gray-700 hover:bg-gray-100'
                      }`}
                    >
                      <span className="truncate flex-1 text-left">{session.title}</span>
                      <button
                        onClick={e => {
                          e.stopPropagation();
                          setMenuOpenId(menuOpenId === session.id ? null : session.id);
                        }}
                        className="opacity-0 group-hover:opacity-100 p-0.5 hover:bg-gray-300 rounded transition-opacity"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z" />
                        </svg>
                      </button>
                    </button>
                  )}

                  {/* Session Menu */}
                  {menuOpenId === session.id && (
                    <div className="absolute right-0 mt-1 w-36 bg-white rounded-lg shadow-lg border border-gray-200 py-1 z-10">
                      <button
                        onClick={() => startEditing(session)}
                        className="w-full px-3 py-1.5 text-sm text-left text-gray-700 hover:bg-gray-100"
                      >
                        Rename
                      </button>
                      <button
                        onClick={() => {
                          setMenuOpenId(null);
                          if (confirm('Delete this conversation?')) {
                            onDeleteSession(session.id);
                          }
                        }}
                        className="w-full px-3 py-1.5 text-sm text-left text-red-600 hover:bg-red-50"
                      >
                        Delete
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          ))
        )}
      </div>

      {/* User Profile */}
      <div className="p-3 border-t border-gray-200" ref={userMenuRef}>
        <div className="relative">
          <button
            onClick={() => setUserMenuOpen(!userMenuOpen)}
            className="w-full flex items-center gap-3 px-2 py-2 rounded-lg hover:bg-gray-100 transition-colors"
          >
            <div className="w-8 h-8 rounded-full bg-gray-900 text-white flex items-center justify-center text-sm font-medium">
              {user?.username.charAt(0).toUpperCase()}
            </div>
            <div className="flex-1 text-left min-w-0">
              <p className="text-sm font-medium text-gray-900 truncate">{user?.username}</p>
            </div>
            <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {/* User Menu */}
          {userMenuOpen && (
            <div className="absolute bottom-full left-0 right-0 mb-1 bg-white rounded-lg shadow-lg border border-gray-200 py-1">
              <button
                onClick={logout}
                className="w-full px-3 py-2 text-sm text-left text-gray-700 hover:bg-gray-100"
              >
                Sign out
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * Group sessions by date (Today, Yesterday, Previous 7 Days, Older)
 */
function groupSessionsByDate(sessions: ChatSession[]): Record<string, ChatSession[]> {
  const groups: Record<string, ChatSession[]> = {};
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  const weekAgo = new Date(today);
  weekAgo.setDate(weekAgo.getDate() - 7);

  for (const session of sessions) {
    const sessionDate = new Date(session.updated_at);
    const sessionDay = new Date(sessionDate.getFullYear(), sessionDate.getMonth(), sessionDate.getDate());

    let group: string;
    if (sessionDay >= today) {
      group = 'Today';
    } else if (sessionDay >= yesterday) {
      group = 'Yesterday';
    } else if (sessionDay >= weekAgo) {
      group = 'Previous 7 Days';
    } else {
      group = 'Older';
    }

    if (!groups[group]) {
      groups[group] = [];
    }
    groups[group].push(session);
  }

  return groups;
}
