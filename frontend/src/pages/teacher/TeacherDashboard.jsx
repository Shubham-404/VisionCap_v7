import { useEffect, useState } from "react";
import { collection, doc, getDoc, getDocs, query, where } from "firebase/firestore";
import { db } from "../../firebase/config";
import { useAuth } from "../../contexts/AuthContext";


const TeacherDashboard = () => {
  const { currentUser } = useAuth();
  const [attendanceLogs, setAttendanceLogs] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchLogs = async () => {
      if (!currentUser) return;

      try {
        // Fetch user data to check role
        const userQuery = query(
          collection(db, 'users'),
          where('uid', '==', currentUser.uid)
        );
        const userSnap = await getDocs(userQuery);

        if (userSnap.empty) {
          console.error('User not found.');
          return;
        }

        const userData = userSnap.docs[0].data();
        if (userData.role !== 'teacher') {
          console.error('Access denied. Not a teacher.');
          return;
        }

        // Fetch attendance logs
        const attendanceSnapshot = await getDocs(collection(db, 'attendance-logs'));
        const logs = [];

        attendanceSnapshot.forEach((doc) => {
          const data = doc.data();
          if (data.students && Array.isArray(data.students)) {
            data.students.forEach((student) => {
              logs.push({
                date: data.date,
                studentName: student.name,
                present: student.present,
                engagement: student.concentrationScore || '-',
              });
            });
          }
        });

        setAttendanceLogs(logs);
      } catch (error) {
        console.error('Error fetching data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchLogs();
  }, [currentUser]);

  if (loading) {
    return (
      <section className="flex justify-center items-center min-h-screen">
        <div className="text-xl font-semibold text-gray-600 animate-pulse">
          Loading dashboard...
        </div>
      </section>
    );
  }

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
              {attendanceLogs.length > 0 ? (
                attendanceLogs.map((log, index) => (
                  <tr key={index} className="border-b">
                    <td className="px-4 py-2">{log.date}</td>
                    <td className="px-4 py-2">{log.studentName}</td>
                    <td
                      className={`px-4 py-2 font-semibold ${log.present === 'true' ? 'text-green-600' : 'text-red-500'
                        }`}
                    >
                      {log.present === 'true' ? 'Present' : 'Absent'}
                    </td>
                    <td className="px-4 py-2">{log.engagement}%</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="4" className="px-4 py-6 text-center text-gray-500">
                    No attendance records found.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
};

export default TeacherDashboard;
