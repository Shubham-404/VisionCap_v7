import React from 'react'; import { Link } from 'react-router-dom';

const NotFound = () => { return ( <div className="min-h-screen flex flex-col items-center justify-center bg-slate-100 text-gray-800 px-4 text-center"> <h1 className="text-5xl font-bold mb-4">404</h1> <p className="text-lg mb-6">Oops! The page you’re looking for doesn’t exist.</p> <Link to="/" className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition" > Back to Home </Link> </div> ); };

export default NotFound;