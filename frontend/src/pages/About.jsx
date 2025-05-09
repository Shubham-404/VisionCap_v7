import React from 'react';

const About = () => {
  return (
    <section className="min-h-screen bg-gradient-to-b from-slate-200 to-slate-100 text-gray-800 px-6">
      <div className="max-w-5xl mx-auto text-center pt-24 pb-20 animate-fadeIn">
        <h1 className="text-5xl font-bold mb-6">About VisionCapture</h1>
        <p className="text-lg text-gray-600 mb-8 max-w-3xl mx-auto">
          VisionCapture is an AI-driven video analytics platform designed to empower classrooms, offices, and public spaces with real-time behavioral insights — all through the lens of any existing camera.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-10 mt-10">
          <div className="bg-white p-6 rounded-xl shadow hover:shadow-md transition text-left">
            <h2 className="text-2xl font-semibold mb-2">Our Mission</h2>
            <p className="text-sm text-gray-600">
              We aim to bring transparency, accountability, and data-driven decision-making to physical and virtual spaces — while upholding the highest standards of privacy and ethical AI.
            </p>
          </div>
          <div className="bg-white p-6 rounded-xl shadow hover:shadow-md transition text-left">
            <h2 className="text-2xl font-semibold mb-2">Our Technology</h2>
            <p className="text-sm text-gray-600">
              Built with edge-AI and privacy-first architecture, VisionCapture delivers accurate insights with minimal latency and no need for intrusive hardware.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default About;
