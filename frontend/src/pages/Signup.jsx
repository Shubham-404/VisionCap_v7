import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate, Link } from 'react-router-dom';

const Signup = () => {
  const [orgName, setOrgName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { signup } = useAuth();
  const navigate = useNavigate();

  const handleSignup = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await signup(email, password);
      navigate('/onboarding');

    } catch (error) {
      setError('Failed to sign up: ' + error.message);

    } finally {
      setLoading(false);

    }
  };

  return (
    <section className="min-h-[90vh] bg-gradient-to-b from-slate-400 to-slate-100 text-gray-800 flex flex-col md:flex-row scroll-smooth">

      <div className="md:w-1/2 relative bg-cover bg-center h-64 md:h-auto" style={{ backgroundImage: 'url(https://images.unsplash.com/photo-1629654297299-c8506221ca97?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80)' }}>
        <div className="absolute inset-0 bg-blue-900/50 flex items-center justify-center">
          <div className="text-center text-white px-6">
            <h2 className="text-3xl md:text-4xl font-bold mb-4 animate-fadeIn">Join the Future</h2>
            <p className="text-lg md:text-xl animate-fadeIn delay-100">Transform spaces with intelligent video analytics.</p>
          </div>
        </div>
      </div>
      <div className="md:w-1/2 flex items-center justify-center p-6 md:p-12">
        <div className="max-w-md w-full bg-white p-8 rounded-xl shadow-lg animate-fadeIn">
          <h2 className="text-3xl font-bold text-center text-gray-800 mb-6">Sign Up</h2>
          {error && (
            <div className="bg-red-100 text-red-700 px-4 py-2 rounded mb-4 text-sm">
              {error}
            </div>

          )}
          <form onSubmit={handleSignup} className="space-y-4">
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-1">Organization Name</label>
              <input
                type="text"
                className="w-full px-3 py-2 border border-gray-400 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600 mb-3"
                placeholder="Organization Name"
                value={orgName}
                onChange={(e) => setOrgName(e.target.value)}
                required
              />
              <label className="block text-sm font-semibold text-gray-700 mb-1">Email</label>
              <input
                type="email"
                className="w-full px-3 py-2 border border-gray-400 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
                placeholder="Email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-1">Password</label>
              <input
                type="password"
                className="w-full px-3 py-2 border border-gray-400 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-8 rounded-lg shadow transition duration-300 hover:scale-105"
            >
              {loading ? 'Signing up...' : 'Sign Up'}
            </button>
          </form>
          <p className="text-sm text-center mt-6 text-gray-600">
            Already have an account?{' '}
            <Link to="/login" className="text-blue-600 hover:underline">
              Sign In
            </Link>
          </p>
        </div>
      </div>
    </section>
  );
};

export default Signup;