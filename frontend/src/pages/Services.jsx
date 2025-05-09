import { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import { doc, getDoc } from 'firebase/firestore';
import { db } from '../firebase/config';
import { updateOrganizationServices } from '../firebase/utils/firestore';
import ModuleCard from '../components/ModuleCard';

function Services() {
  const { userProfile, loading: authLoading } = useAuth();
  const navigate = useNavigate();
  const [selectedModules, setSelectedModules] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const attend = 'https://www.edecofy.com/blog/wp-content/uploads/2021/03/Student-Attendance-Automation-1024x538.png';
  const face_rec = 'https://empmonitor.com/blog/wp-content/uploads/2024/11/What-Is-a-Facial-Recognition-Attendance-System-1024x576.webp';
  const behave_feed = 'https://businesspedia.in/wp-content/uploads/2021/12/Group-23.webp';
  const faculty_ins = 'https://ahduni.edu.in/site/assets/files/7325/760_x_540_amsom_faculty.1400x0.webp';
  const student_ins = 'https://newmetrics.com/files/uploads/2023/08/Student-Experience-Cover-2-768x307.jpg';
  const heat_map = 'https://powerslides.com/wp-content/uploads/2022/01/PowerPoint-Heatmap-Template-4.png';

  const imageUrls = [attend, face_rec, behave_feed, faculty_ins, student_ins, heat_map];

  const getRandomImage = () => {
    return imageUrls[Math.floor(Math.random() * imageUrls.length)];
  };

  const modules = [
    { title: 'Attendance Tracking', description: 'Monitor attendance using AI-powered face recognition.', imgUrl: imageUrls[0] },
    { title: 'Face Recognition', description: 'Identify individuals in video feeds for security and analytics.', imgUrl: imageUrls[1] },
    { title: 'Behavior Analysis', description: 'Analyze behavior patterns in classrooms or offices.', imgUrl: imageUrls[2] },
    { title: 'Faculty Insights', description: 'Generate detailed reports for teachers and faculty.', imgUrl: imageUrls[3] },
    { title: 'Student Insights', description: 'Provide personalized insights for student engagement.', imgUrl: imageUrls[4] },
    { title: 'Heatmaps', description: 'Visualize organization-wide usage and activity patterns.', imgUrl: imageUrls[5] },
  ];

  useEffect(() => {
    // Debug: Log userProfile to diagnose role issue
    console.log('userProfile:', userProfile);

    if (authLoading) return; // Wait for auth to load

    const fetchServices = async () => {
      if (userProfile && userProfile.orgIds?.[0]) {
        try {
          const orgDoc = await getDoc(doc(db, 'organizations', userProfile.orgIds[0]));
          if (orgDoc.exists()) {
            setSelectedModules(orgDoc.data().activeServices || []);
          } else {
            setError('Organization not found. Please complete onboarding.');
          }
        } catch (err) {
          setError('Failed to load services: ' + err.message);
        }
      } else if (userProfile && !userProfile.orgIds?.[0]) {
        setError('No organization assigned. Please complete onboarding.');
      }
      setLoading(false);
    };

    fetchServices();

    // Handle unauthenticated or incomplete userProfile
    if (!userProfile) {
      setError('Please log in to select services.');
    } else if (!userProfile.role) {
      setError('No role assigned. Please complete onboarding.');
      navigate('/onboarding');
    }
  }, [userProfile, authLoading, navigate]);

  const handleToggle = (title) => {
    const newModules = selectedModules.includes(title)
      ? selectedModules.filter((m) => m !== title)
      : [...selectedModules, title];
    setSelectedModules(newModules);
  };

  const handleSubmit = async () => {
    if (!userProfile || !userProfile.orgIds?.[0]) {
      setError(`No organization assigned. Please complete onboarding.`);
      return;
    }
    setError('');
    setSuccess('');
    try {
      await updateOrganizationServices(userProfile.orgIds[0], selectedModules);
      setSuccess('Services updated successfully!');
      setTimeout(() => setSuccess(''), 3000); // Clear success message after 3s
    } catch (err) {
      setError('Failed to update services: ' + err.message);
    }
  };

  // Normalize role to handle case sensitivity
  const isOrganization = userProfile && userProfile.role && userProfile.role.toLowerCase() === 'organization';

  if (authLoading || loading) {
    return (
      <section className="min-h-screen bg-gradient-to-b from-slate-50 to-white text-gray-800 px-6 scroll-smooth">
        <div className="max-w-6xl mx-auto py-12 animate-fadeIn">
          <p className="text-lg text-gray-600">Loading services...</p>
        </div>
      </section>
    );
  }

  return (
    <section className="min-h-screen bg-gradient-to-b from-slate-50 to-white text-gray-800 px-6 scroll-smooth">
      <div className="max-w-6xl mx-auto py-12 animate-fadeIn">
        <h1 className="text-3xl font-bold mb-4">AI Services</h1>
        <p className="text-lg text-gray-600 mb-8">
          {isOrganization
            ? 'Select the AI-powered video analytics services for your organization.'
            : 'Explore our AI-powered video analytics services.'}
        </p>
        {error && (
          <div className="bg-red-100 text-red-700 px-4 py-2 rounded mb-4 text-sm">{error}</div>
        )}
        {success && (
          <div className="bg-green-100 text-green-700 px-4 py-2 rounded mb-4 text-sm">{success}</div>
        )}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8 mb-8">
          {modules.map((module) => (
            <ModuleCard
              key={module.title}
              title={module.title}
              description={module.description}
              imgUrl={module.imgUrl}
              isSelected={selectedModules.includes(module.title)}
              onToggle={isOrganization ? () => handleToggle(module.title) : undefined}
            />
          ))}
        </div>
        <div className="flex justify-center space-x-4">
          {isOrganization && (
            <>
              <button
                onClick={handleSubmit}
                className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-8 rounded-lg shadow transition duration-300 hover:scale-105"
              >
                Submit
              </button>
              <button
                onClick={() => navigate('/admin/dashboard')}
                className="bg-gray-600 hover:bg-gray-700 text-white font-semibold py-3 px-8 rounded-lg shadow transition duration-300 hover:scale-105"
              >
                Back to Dashboard
              </button>
            </>
          )}
          {!isOrganization && (
            <button
              onClick={() => navigate(userProfile ? '/onboarding' : '/login')}
              className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-8 rounded-lg shadow transition duration-300 hover:scale-105"
            >
              {userProfile ? 'Complete Onboarding to Select Services' : 'Log In to Select Services'}
            </button>
          )}
        </div>
      </div>
    </section>
  );
}

export default Services;