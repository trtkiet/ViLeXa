/**
 * Registration page
 */

import { useNavigate } from 'react-router-dom';
import { RegisterForm } from '../components/auth/RegisterForm';
import { useAuth } from '../contexts/AuthContext';
import { useEffect } from 'react';

export function RegisterPage() {
  const navigate = useNavigate();
  const { isAuthenticated } = useAuth();

  // Redirect if already authenticated
  useEffect(() => {
    if (isAuthenticated) {
      navigate('/', { replace: true });
    }
  }, [isAuthenticated, navigate]);

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
            Create your account
          </h1>

          <RegisterForm />
        </div>
      </div>
    </div>
  );
}
