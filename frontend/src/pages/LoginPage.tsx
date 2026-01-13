/**
 * Login page
 */

import { useNavigate, useLocation } from 'react-router-dom';
import { LoginForm } from '../components/auth/LoginForm';
import { useAuth } from '../contexts/AuthContext';
import { useEffect } from 'react';

export function LoginPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { isAuthenticated } = useAuth();

  // Get the intended destination from location state
  const from = location.state?.from?.pathname || '/';
  const justRegistered = location.state?.registered === true;

  // Redirect if already authenticated
  useEffect(() => {
    if (isAuthenticated) {
      navigate(from, { replace: true });
    }
  }, [isAuthenticated, navigate, from]);

  const handleSuccess = () => {
    navigate(from, { replace: true });
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 px-4 py-12">
      <div className="w-full max-w-sm">
        {/* Logo */}
        <div className="flex items-center justify-center gap-2 mb-8">
          <span className="bg-gray-900 text-white px-2.5 py-1 rounded text-lg font-bold">VA</span>
          <span className="text-xl font-semibold text-gray-900">Law Assistant</span>
        </div>

        {/* Card */}
        <div className="bg-white p-8 rounded-xl shadow-sm border border-gray-200">
          <h1 className="text-xl font-semibold text-gray-900 mb-6 text-center">
            Sign in to your account
          </h1>

          {/* Success message after registration */}
          {justRegistered && (
            <div className="mb-6 p-3 bg-green-50 border border-green-200 rounded-lg text-green-600 text-sm">
              Account created successfully! Please sign in.
            </div>
          )}

          <LoginForm onSuccess={handleSuccess} />
        </div>
      </div>
    </div>
  );
}
