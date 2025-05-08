<<<<<<< HEAD
import { React, useRef, userEffect, useState } from 'react';
import { useAuth } from '../contexts/AuthContext'; // Adjust path if needed

import { Link } from "react-router-dom";
=======
import { useState } from 'react';
import { Link } from "react-router-dom";
import { useAuth } from '../contexts/AuthContext';
>>>>>>> daaf47cbbe7b82a32e589cda4ed92310382d84ad

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

<<<<<<< HEAD
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
=======
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
>>>>>>> daaf47cbbe7b82a32e589cda4ed92310382d84ad
          )}
        </nav>
      </div>
    </header>
  );
};

export default Header;
