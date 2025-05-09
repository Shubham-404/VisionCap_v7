import { Link, useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { doc, getDoc } from 'firebase/firestore';
import { db } from '../firebase/config';
import { saveDemoRequest } from '../firebase/utils/firestore';

const Home = () => {
  const { currentUser } = useAuth();
  const navigate = useNavigate();
  const [role, setRole] = useState(null);
  const [loading, setLoading] = useState(true);
  const [formData, setFormData] = useState({
    orgName: '',
    email: '',
    videoType: 'loksabha',
    service: 'attendance',
    customVideoUrl: ''

  });
  const [formError, setFormError] = useState('');
  const [formSuccess, setFormSuccess] = useState('');

  useEffect(() => {
    const fetchRole = async () => {
      if (currentUser) {
        try {
          const userDoc = await getDoc(doc(db, 'users', currentUser.uid));
          if (userDoc.exists()) {
            setRole(userDoc.data().role);
            setFormData((prev) => ({ ...prev, email: userDoc.data().email }));
          }
        } catch (err) {
          console.error('Error fetching user role:', err);

        }
      }
      setLoading(false);
    };
    fetchRole();
  }, [currentUser]);

  const handleGoToDashboard = () => {
    if (role === 'student') navigate('/dashboard');
    else if (role === 'teacher') navigate('/teacher/dashboard');
    else if (role === 'organization') navigate('/services');

  };

  const handleFormChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleFormSubmit = async (e) => {
    e.preventDefault();
    setFormError('');
    setFormSuccess('');
    if (!formData.orgName || !formData.email) {
      setFormError('Organization name and email are required.');
      return;

    }
    if (formData.videoType === 'custom' && !formData.customVideoUrl) {
      setFormError('Custom video URL is required for custom demos.');
      return;

    }
    try {
      await saveDemoRequest(formData);
      setFormSuccess('Demo request submitted successfully!');
      setFormData({
        orgName: '',
        email: currentUser ? formData.email : '',
        videoType: 'loksabha',
        service: 'attendance',
        customVideoUrl: ''

      });
    } catch (err) {
      setFormError('Failed to submit demo request: ' + err.message);

    }
  };

  return (

    <section className="min-h-screen bg-gradient-to-b from-slate-200 to-slate-100 text-gray-800 px-6 scroll-smooth">
      <div className="max-w-6xl mx-auto text-center pt-24 pb-20 animate-fadeIn">
        <h1 className="text-5xl font-bold mb-6 leading-tight">
          AI-Powered Video Analytics for Smarter Spaces
        </h1>
        <p className="text-lg text-gray-600 mb-8 max-w-3xl mx-auto">
          Transform any camera-enabled environment — classrooms, offices, retail, or online sessions — with intelligent behavior tracking and real-time engagement analytics.
        </p>
        {loading ? null : currentUser && role ? (
          <button
            onClick={handleGoToDashboard}
            className="bg-purple-600 hover:bg-purple-700 text-white font-semibold py-3 px-8 rounded-lg shadow-lg transition duration-300 hover:scale-105"
          >
            Go to Dashboard
          </button>

        ) : (
          <div className="flex flex-col sm:flex-row justify-center gap-4">
            <Link
              to="/login"
              className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-8 rounded-lg shadow transition duration-300 hover:scale-105"
            >
              Login
            </Link>
            <Link
              to="mailto:contact@yourcompany.com"
              className="bg-green-600 hover:bg-green-700 text-white font-semibold py-3 px-8 rounded-lg shadow transition duration-300 hover:scale-105"
            >
              Partner With Us
            </Link>
          </div>

        )}
      </div>

      <h1 className='text-center text-2xl font-semibold'>Why choose VisionCapture?</h1>
      <div className="max-w-6xl mx-auto py-8 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8">
        {[
          {
            icon: '🎥',
            title: 'Compatible with Any Camera',
            desc: 'Easily integrate with standard webcams or CCTV for smart analytics.',

          },
          {
            icon: '📊',
            title: 'Behavior & Emotion Detection',
            desc: 'Track attention, mood, posture, and anomalies in real-time.',

          },
          {
            icon: '📁',
            title: 'Session Summaries',
            desc: 'Export attendance, engagement, and event logs with one click.',

          },
          {
            icon: '⚙️',
            title: 'Flexible API Integration',
            desc: 'Seamlessly plug into your LMS, HR tools, or surveillance systems.',

          },

        ].map(({ icon, title, desc }, idx) => (
          <div
            key={idx}
            className="bg-white p-6 rounded-xl shadow text-center hover:shadow-lg transition duration-300 hover:scale-[1.03]"
          >
            <div className="text-3xl mb-3">{icon}</div>
            <h3 className="text-lg font-semibold">{title}</h3>
            <p className="text-sm text-gray-600 mt-2">{desc}</p>
          </div>

        ))}
      </div>

      <div className="min-h-[80%] rounded-3xl overflow-hidden bg-gradient-to-b from-slate-100 to-slate-100 text-gray-800 flex flex-col md:flex-row scroll-smooth">

        <div className="md:w-1/2 relative bg-cover bg-center h-64 md:h-auto" style={{ backgroundImage: 'url(https://images.unsplash.com/photo-1629654297299-c8506221ca97?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80)' }}>
          <div className="absolute inset-0 bg-blue-900/50 flex items-center justify-center">
            <div className="text-center text-white px-6">
              <h2 className="text-3xl md:text-4xl font-bold mb-4 animate-fadeIn">Get a demo of our service.</h2>
              <p className="text-lg md:text-xl animate-fadeIn delay-100 italic">"Pehle istemaal karein, phir vishvaas karein."</p>
            </div>
          </div>
        </div>
        <section className="md:w-1/2 p-8 flex flex-col justify-center">
          <h2 className="text-3xl font-bold text-center mb-8">Request a Demo</h2>
          <form
            onSubmit={handleFormSubmit}
            className="w-sm mx-auto bg-white p-6 rounded-xl shadow-lg"
          >
            {formError && (
              <div className="bg-red-100 text-red-700 px-4 py-2 rounded mb-4 text-sm">
                {formError}
              </div>
            )}
            {formSuccess && (
              <div className="bg-green-100 text-green-700 px-4 py-2 rounded mb-4 text-sm">
                {formSuccess}
              </div>
            )}

            {/* Organization Name */}
            <div className="mb-4">
              <label className="block text-sm font-semibold text-gray-700 mb-1">
                Organization Name
              </label>
              <input
                type="text"
                name="orgName"
                value={formData.orgName}
                onChange={handleFormChange}
                className="w-full px-3 py-2 border border-gray-400 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
                required
              />
            </div>

            {/* Email */}
            <div className="mb-4">
              <label className="block text-sm font-semibold text-gray-700 mb-1">
                Email
              </label>
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleFormChange}
                className="w-full px-3 py-2 border border-gray-400 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
                required
                disabled={currentUser}
              />
            </div>

            {/* Video Type */}
            <div className="mb-4">
              <label className="block text-sm font-semibold text-gray-700 mb-1">
                Demo Video Type
              </label>
              <select
                name="videoType"
                value={formData.videoType}
                onChange={handleFormChange}
                className="w-full px-3 py-2 border border-gray-400 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
              >
                <option value="loksabha">Lok Sabha (Demo)</option>
                <option value="custom">Custom Video</option>
              </select>
            </div>

            {/* Service */}
            <div className="mb-4">
              <label className="block text-sm font-semibold text-gray-700 mb-1">
                Service
              </label>
              <select
                name="service"
                value={formData.service}
                onChange={handleFormChange}
                className="w-full px-3 py-2 border border-gray-400 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
              >
                <option value="attendance">Attendance Tracking</option>
                <option value="faceRecognition">Face Recognition</option>
                <option value="behavior">Behavior Analysis</option>
              </select>
            </div>

            {/* Custom Video URL (conditional) */}
            {formData.videoType === 'custom' && (
              <div className="mb-4">
                <label className="block text-sm font-semibold text-gray-700 mb-1">
                  Custom Video URL
                </label>
                <input
                  type="url"
                  name="customVideoUrl"
                  value={formData.customVideoUrl}
                  onChange={handleFormChange}
                  className="w-full px-3 py-2 border border-gray-400 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
                  placeholder="https://storage.googleapis.com/..."
                  required
                />
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-8 rounded-lg shadow transition duration-300 hover:scale-105"
            >
              Submit Demo Request
            </button>
          </form>
        </section>

      </div>



      <div className="bg-slate-100 py-16 animate-fadeIn delay-200">
        <div className="max-w-5xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-4">Why Choose Our Platform?</h2>
          <p className="text-gray-600 mb-10 max-w-2xl mx-auto">
            Our edge-AI technology delivers real-time, privacy-conscious insights from any environment — without the need for invasive hardware or manual supervision.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              { title: '⚡ Industry-Agnostic', desc: 'From classrooms to call centers — our tech fits everywhere.' },
              { title: '🔐 Privacy Compliant', desc: 'Built with GDPR and FERPA standards in mind.' },
              { title: '📡 Edge AI Capable', desc: 'Run analysis locally with low-latency performance.' },

            ].map(({ title, desc }, idx) => (
              <div key={idx} className="p-6 bg-white rounded-xl shadow hover:shadow-md transition">
                <h4 className="text-lg font-semibold mb-2">{title}</h4>
                <p className="text-sm text-gray-600">{desc}</p>
              </div>

            ))}
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto py-16 px-4 animate-fadeInUp">
        <h2 className="text-3xl font-bold text-center mb-10">Versatile Across Industries</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-8">
          {[
            { title: '🏫 Smart Classrooms', desc: 'Automated engagement tracking and attendance.' },
            { title: '🏢 Corporate Training', desc: 'Measure participation and focus during meetings or e-learning.' },
            { title: '🛍️ Retail & Public Spaces', desc: 'Analyze customer emotions, footfall, and dwell time patterns.' },

          ].map(({ title, desc }, idx) => (
            <div key={idx} className="bg-white p-6 rounded-xl shadow text-center hover:shadow-lg transition">
              <h4 className="text-lg font-semibold mb-2">{title}</h4>
              <p className="text-sm text-gray-600">{desc}</p>
            </div>

          ))}
        </div>
      </div>

      <div className="bg-purple-700 py-12 text-center text-white animate-fadeInUp delay-300">
        <h2 className="text-3xl font-bold mb-4">Empower Your Space with Video Intelligence</h2>
        <p className="text-md mb-6">AI-driven insights that scale with your vision.</p>
        <Link
          to="/login"
          className="bg-white text-purple-700 font-semibold py-2 px-6 rounded-lg shadow hover:bg-gray-100 transition"
        >
          Get Started
        </Link>
      </div>

      <div className="py-12 text-center text-gray-500 text-sm">
        <p>Trusted by forward-thinking educators, enterprises, and developers worldwide.</p>
      </div>
    </section>
  );
};

export default Home;