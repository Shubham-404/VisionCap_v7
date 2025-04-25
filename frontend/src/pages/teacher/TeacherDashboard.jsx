const AdminDashboard = () => {
  return (
    <section className="container mx-auto px-4 py-6">
      <h2 className="text-2xl font-semibold text-gray-800 mb-4">Teacher Dashboard</h2>

      {/* Class Controls */}
      <div className="bg-white rounded-xl shadow p-6 mb-6 space-y-4">
        <h3 className="text-lg font-semibold text-gray-700">Class Controls</h3>
        <div className="flex gap-4 flex-wrap">
          <button className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition">
            Start Class
          </button>
          <button className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition">
            Start Analysis
          </button>
          <button className="bg-red-500 text-white px-4 py-2 rounded-lg hover:bg-red-600 transition">
            Stop Analysis
          </button>
        </div>
      </div>

      {/* Attendance Logs */}
      <div className="bg-white rounded-xl shadow p-6">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">Past Attendance Logs</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm text-left text-gray-600">
            <thead>
              <tr className="bg-gray-100 text-gray-700">
                <th className="px-4 py-2 font-medium">Date</th>
                <th className="px-4 py-2 font-medium">Student</th>
                <th className="px-4 py-2 font-medium">Status</th>
                <th className="px-4 py-2 font-medium">Engagement</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b">
                <td className="px-4 py-2">2025-04-23</td>
                <td className="px-4 py-2">Aarav Mehta</td>
                <td className="px-4 py-2 text-green-600 font-semibold">Present</td>
                <td className="px-4 py-2">88%</td>
              </tr>
              <tr className="border-b">
                <td className="px-4 py-2">2025-04-23</td>
                <td className="px-4 py-2">Diya Sharma</td>
                <td className="px-4 py-2 text-red-500 font-semibold">Absent</td>
                <td className="px-4 py-2">-</td>
              </tr>
              {/* More rows here... */}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
};

export default AdminDashboard;
