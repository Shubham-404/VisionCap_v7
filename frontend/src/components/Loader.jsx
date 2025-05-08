import React from 'react';

const Loader = () => { return (<div className="fixed inset-0 bg-white flex items-center justify-center z-50"> <div className="animate-spin rounded-full h-10 w-10 border-t-4 border-purple-600"></div> </div>); };

export default Loader;