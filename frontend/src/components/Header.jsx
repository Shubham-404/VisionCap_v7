import { useState } from 'react';
import { Link } from "react-router-dom";
import { useAuth } from '../contexts/AuthContext';

const Header = () => {
  const [showUserDetails, setShowUserDetails] = useState(false);
  const { currentUser, logout } = useAuth();

  const handleLogout = async () => {
    try {
      await logout();
    } catch (error) {
      console.error("Logout failed:", error);
    }
  };

  const toggleUserInfo = () => setShowUserDetails(prev => !prev);

  return (
    <header className="bg-white shadow-md sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4 flex justify-between items-center">
        <Link to="/" className="text-2xl font-bold text-indigo-600 tracking-tight">
          VisionCapture
        </Link>
        <nav className="flex gap-6 items-center text-sm font-medium text-gray-700">
          <Link to="/" className="hover:text-indigo-600">Home</Link>
          <Link to="/features" className="hover:text-indigo-600">Features</Link>
          <Link to="/about" className="hover:text-indigo-600">About</Link>
          <Link to="/contact" className="hover:text-indigo-600">Contact</Link>
          {currentUser ? (
            <div className="flex items-center gap-4">
              <button onClick={toggleUserInfo} className="hover:text-indigo-600">
                {showUserDetails
                  ? `${currentUser.displayName || currentUser.email}`
                  : `${(currentUser.displayName || currentUser.email).split('@')[0].slice(0, 6)}...`}
              </button>
              <button
                onClick={handleLogout}
                className="bg-red-500 hover:bg-red-600 text-white py-1 px-4 rounded-lg"
              >
                Logout
              </button>
            </div>
          ) : (
            <Link
              to="/login"
              className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-1 rounded-lg"
            >
              Login
            </Link>
          )}
        </nav>
      </div>
    </header>
  );
};

export default Header;
