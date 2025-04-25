import { React, useRef, userEffect, useState } from 'react';
import { useAuth } from '../contexts/AuthContext'; // Adjust path if needed

import { Link } from "react-router-dom";

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

  const toggleUserInfo = () => {
    setShowUserDetails(prev => !prev);
  }

  return (
    <header className="bg-white shadow-md">
      <div className="container mx-auto px-4 py-3 flex justify-between items-center">
        <Link to="/" className="text-xl font-bold text-blue-600">
          Vision Caputure
        </Link>
        <nav className="hidden md:flex space-x-6 items-center">
          <Link to="/" className="text-gray-700 hover:text-blue-600 font-medium">
            Dashboard
          </Link>
          
          {!currentUser ? (
            <div className='flex gap-10'>
              <Link to="/login" className="!text-white text-lg p-3 py-1 rounded-xl bg-[#398ae6] hover:bg-[#5788af]">Login</Link>
            </div>
          ) : (
            <div className='flex items-center gap-5 text-white'>
              {/* for identifing student and organizer */}
              <Link to="/profile"
                onClick={toggleUserInfo}
                className="cursor-pointer text-lg text-black hover:text-[#E63946] transition"
              >
                {showUserDetails
                  ? <>

                    {currentUser.displayName || currentUser.name} {currentUser.email}
                  </>
                  : `${currentUser.displayName || currentUser.email}`.toUpperCase().slice(0, 7) + "..."}
              </Link>

              <button onClick={handleLogout} className="text-white text-lg p-3 py-1 rounded-sm bg-[#E63946] hover:bg-[#F63956]">Logout</button>
            </div>
          )}
        </nav>
      </div>
    </header>
  );
};

export default Header;
