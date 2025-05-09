import { useAuth } from '../../contexts/AuthContext';

function StudentProfile() {
  const { userProfile } = useAuth();

  return (
    <section className="min-h-screen bg-gradient-to-b from-slate-50 to-white text-gray-800 px-6 scroll-smooth">
      <div className="max-w-6xl mx-auto py-12 animate-fadeIn">
        <h1 className="text-3xl font-bold mb-4">Student Profile</h1>
        <div className="bg-white p-6 rounded-xl shadow hover:shadow-lg transition duration-300">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Profile Details</h3>
          <p className="text-sm text-gray-600"><strong>Name:</strong> {userProfile?.displayName || 'N/A'}</p>
          <p className="text-sm text-gray-600"><strong>Email:</strong> {userProfile?.email || 'N/A'}</p>
          <p className="text-sm text-gray-600"><strong>Role:</strong> {userProfile?.role || 'N/A'}</p>
          <p className="text-sm text-gray-600"><strong>Organization:</strong> {userProfile?.orgIds?.[0] || 'N/A'}</p>
        </div>
      </div>
    </section>
  );
}

export default StudentProfile;