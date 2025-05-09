import { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import { doc, getDoc } from 'firebase/firestore';
import { db } from '../../firebase/config';

const TeacherDashboard = () => {
  const { userProfile } = useAuth();
  const navigate = useNavigate();
  const [activeServices, setActiveServices] = useState([]);
  const [semester, setSemester] = useState('');
  const [room, setRoom] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const modules = [
    { title: 'Attendance Tracking', description: 'Monitor attendance using AI-powered face recognition.' },
    { title: 'Face Recognition', description: 'Identify individuals in video feeds for security and analytics.' },
    { title: 'Behavior Analysis', description: 'Analyze behavior patterns in classrooms or offices.' },
    { title: 'Faculty Insights', description: 'Generate detailed reports for teachers and faculty.' },
    { title: 'Student Insights', description: 'Provide personalized insights for student engagement.' },
    { title: 'Heatmaps', description: 'Visualize organization-wide usage and activity patterns.' },
  ];

  useEffect(() => {
    const fetchServices = async () => {
      if (userProfile && userProfile.orgIds?.[0]) {
        try {
          const orgDoc = await getDoc(doc(db, 'organizations', userProfile.orgIds[0]));
          if (orgDoc.exists()) {
            setActiveServices(orgDoc.data().activeServices || []);
          }
        } catch (err) {
          setError('Failed to load services: ' + err.message);
        }
      }
      setLoading(false);
    };
    fetchServices();
  }, [userProfile]);

  const handleStartClass = () => {
    if (!semester || !room) {
      setError('Please enter semester and room number.');
      return;
    }
    alert(`Starting class for Semester ${semester}, Room ${room}`);
    // Trigger AI analysis here, e.g., send to backend for processing
  };

  if (loading) {
    return (
      <section className="min-h-screen bg-gradient-to-b from-slate-50 to-white text-gray-800 px-6 scroll-smooth">
        <div className="max-w-6xl mx-auto py-12 animate-fadeIn">
          <p className="text-lg text-gray-600">Loading dashboard...</p>
        </div>
      </section>
    );
  }

  return (
    <section className="min-h-screen bg-gradient-to-b from-slate-50 to-white text-gray-800 px-6 scroll-smooth">
      <div className="max-w-6xl mx-auto py-12 animate-fadeIn">
        <h1 className="text-3xl font-bold mb-4">Faculty Dashboard</h1>
        <p className="text-lg text-gray-600 mb-8">Start a class and access AI analytics services.</p>
        {error && (
          <div className="bg-red-100 text-red-700 px-4 py-2 rounded mb-4 text-sm">{error}</div>
        )}
        <div className="bg-white p-6 rounded-xl shadow-lg mb-8">
          <h2 className="text-xl font-semibold mb-4">Start Class</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-1">Semester</label>
              <input
                type="text"
                value={semester}
                onChange={(e) => setSemester(e.target.value)}
                className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-600"
                placeholder="e.g., 5"
              />
            </div>
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-1">Room No.</label>
              <input
                type="text"
                value={room}
                onChange={(e) => setRoom(e.target.value)}
                className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-600"
                placeholder="e.g., LH-101"
              />
            </div>
          </div>
          <button
            onClick={handleStartClass}
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg shadow transition duration-300 hover:scale-105"
          >
            Start Class
          </button>
        </div>
        <div className="bg-white p-6 rounded-xl shadow-lg">
          <h2 className="text-xl font-semibold mb-4">Available Services</h2>
          {activeServices.length > 0 ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {modules
                .filter((module) => activeServices.includes(module.title))
                .map((module) => (
                  <div
                    key={module.title}
                    className="p-4 bg-gray-50 rounded-lg shadow hover:shadow-md transition duration-300"
                  >
                    <h3 className="text-lg font-semibold text-gray-800">{module.title}</h3>
                    <p className="text-sm text-gray-600">{module.description}</p>
                    <button
                      onClick={() => alert(`Running ${module.title} analysis`)}
                      className="mt-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-1 px-4 rounded-lg shadow transition duration-300 hover:scale-105"
                    >
                      Run Analysis
                    </button>
                  </div>
                ))}
            </div>
          ) : (
            <p className="text-sm text-gray-600">No services enabled by your organization.</p>
          )}
        </div>
      </div>
    </section>
  );
};

export default TeacherDashboard;