import { useAuth } from '../../contexts/AuthContext';

function StudentDashboard() {
  const { userProfile } = useAuth();

  return (
    <section className="min-h-screen bg-gradient-to-b from-slate-50 to-white text-gray-800 px-6 scroll-smooth">
      <div className="max-w-6xl mx-auto py-12 animate-fadeIn">
        <h1 className="text-3xl font-bold mb-4">Student Dashboard</h1>
        <p className="text-lg text-gray-600 mb-8">
          Welcome, {userProfile?.displayName || 'Student'}! View your personalized insights and attendance records.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
          <div className="bg-white p-6 rounded-xl shadow hover:shadow-lg transition duration-300 hover:scale-[1.03]">
            <h3 className="text-lg font-semibold text-gray-800">Attendance</h3>
            <p className="text-sm text-gray-600 mt-2">Track your attendance records and trends.</p>
          </div>
          <div className="bg-white p-6 rounded-xl shadow hover:shadow-lg transition duration-300 hover:scale-[1.03]">
            <h3 className="text-lg font-semibold text-gray-800">Insights</h3>
            <p className="text-sm text-gray-600 mt-2">View personalized behavior and engagement insights.</p>
          </div>
          <div className="bg-white p-6 rounded-xl shadow hover:shadow-lg transition duration-300 hover:scale-[1.03]">
            <h3 className="text-lg font-semibold text-gray-800">Reports</h3>
            <p className="text-sm text-gray-600 mt-2">Access detailed session summaries.</p>
          </div>
        </div>
      </div>
    </section>
  );
}

export default StudentDashboard;