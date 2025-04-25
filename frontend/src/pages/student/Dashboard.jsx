const StudentDashboard = () => {
  return (
    <section className=" mx-auto px-4 py-6">
      <h2 className="text-2xl font-semibold text-gray-800 mb-4">
        Welcome, Student 👋
      </h2>

      <div className="bg-white rounded-xl shadow p-6">
        <p className="text-gray-600">
          Here’s a summary of your attendance and engagement for today:
        </p>

        <div className="grid grid-cols-2 md:grid-cols-2 gap-6 mt-6">
          <div className="bg-blue-100 text-blue-800 p-6 rounded-lg">
            <p className="text-sm font-medium">Attendance Status</p>
            <h3 className="text-2xl font-bold mt-2">Present ✅</h3>
            <p className="text-xs mt-1 text-blue-600">9:05 AM - 3:45 PM</p>
          </div>

          <div className="bg-green-100 text-green-800 p-6 rounded-lg">
            <p className="text-sm font-medium">Engagement Score</p>
            <h3 className="text-2xl font-bold mt-2">82%</h3>
            <p className="text-xs mt-1 text-green-600">Mostly attentive</p>
          </div>
        </div>

        <div className="mt-8">
          <h4 className="text-lg font-semibold mb-2">Weekly Attendance Overview</h4>
          <div className="bg-gray-100 rounded-lg h-32 flex items-center justify-center text-gray-500">
            {/* Placeholder for future chart */}
            📊 Graph coming soon...
          </div>
        </div>
      </div>
    </section>
  );
};

export default StudentDashboard;
