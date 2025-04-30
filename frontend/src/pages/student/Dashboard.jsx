import { useEffect, useState } from "react";
import { collection, doc, getDoc, getDocs, query, where } from "firebase/firestore";
import { db } from "../../firebase/config";
import { useAuth } from "../../contexts/AuthContext";

const StudentDashboard = () => {
  const { currentUser } = useAuth();
  const [studentData, setStudentData] = useState(null);
  const [attendanceStatus, setAttendanceStatus] = useState(null);
  const [attendanceTime, setAttendanceTime] = useState(null);
  const [engagementScore, setEngagementScore] = useState(null);
  const [engagementNote, setEngagementNote] = useState(null);
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
        if (userData.role === 'teacher') {
          console.error('Access denied. Not a student.');
          return <Navigate to="/teacher/dashboard" />;;
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

    const fetchDashboardData = async () => {
      if (!currentUser) return;

      try {
        // Fetch user basic profile
        const userDocRef = doc(db, "users", currentUser.uid);
        const userDocSnap = await getDoc(userDocRef);

        if (userDocSnap.exists()) {
          const userData = userDocSnap.data();
          setStudentData(userData);

          const userUsn = userData.Usn;

          // Fetch latest attendance
          const attendanceQuery = query(
            collection(db, "attendance-logs"),
            where("Usn", "==", userUsn)
          );
          const attendanceSnap = await getDocs(attendanceQuery);

          let latestAttendance = null;
          attendanceSnap.forEach((doc) => {
            const data = doc.data();
            if (!latestAttendance || data.timestamp > latestAttendance.timestamp) {
              latestAttendance = data;
            }
          });

          if (latestAttendance && Array.isArray(latestAttendance.students)) {
            const matchedStudent = latestAttendance.students.find(
              (student) => student.usn === userUsn
            );

            if (matchedStudent) {
              setAttendanceStatus(matchedStudent.present === true ? "Present" : "Absent");
              setAttendanceTime(new Date(latestAttendance.timestamp?.seconds * 1000).toLocaleString());
            }
          }

          // Fetch engagement reports
          const engagementQuery = query(
            collection(db, "engagement-reports"),
            where("Usn", "==", userUsn)
          );
          const engagementSnap = await getDocs(engagementQuery);

          let totalScore = 0;
          let reportsCount = 0;

          engagementSnap.forEach((doc) => {
            const data = doc.data();
            if (data.behaviourScore !== undefined) {
              totalScore += data.behaviourScore;
              reportsCount += 1;
            }
          });

          if (reportsCount > 0) {
            setEngagementScore((totalScore / reportsCount).toFixed(2));
            setEngagementNote("Based on recent sessions");
          } else {
            setEngagementScore("N/A");
            setEngagementNote("No engagement reports yet");
          }

        } else {
          console.warn("No such student document!");
        }

      } catch (error) {
        console.error("Error fetching dashboard data:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, [currentUser]);

  if (loading) {
    return <p className="text-center py-10 text-gray-500">Loading dashboard...</p>;
  }

  if (!studentData) {
    return <p className="text-center py-10 text-red-500">No dashboard data found.</p>;
  }

  return (
    <section className="mx-auto px-4 py-6">
      <h2 className="text-2xl font-semibold text-gray-800 mb-4">
        Welcome, {studentData.name?.split(" ")[0] || "Student"} 👋
      </h2>

      <div className="bg-white rounded-xl shadow p-6">
        <p className="text-gray-600">
          Here’s a summary of your attendance and engagement:
        </p>

        <div className="grid grid-cols-2 md:grid-cols-2 gap-6 mt-6">
          <div className="bg-blue-100 text-blue-800 p-6 rounded-lg">
            <p className="text-sm font-medium">Attendance Status</p>
            <h3 className="text-2xl font-bold mt-2">
              {attendanceStatus || "Unknown"}
            </h3>
            <p className="text-xs mt-1 text-blue-600">
              {attendanceTime || "N/A"}
            </p>
          </div>

          <div className="bg-green-100 text-green-800 p-6 rounded-lg">
            <p className="text-sm font-medium">Engagement Score</p>
            <h3 className="text-2xl font-bold mt-2">
              {engagementScore}
            </h3>
            <p className="text-xs mt-1 text-green-600">
              {engagementNote}
            </p>
          </div>
        </div>

        <div className="mt-8">
          <h4 className="text-lg font-semibold mb-2">Weekly Attendance Overview</h4>
          <div className="bg-gray-100 rounded-lg h-32 flex items-center justify-center text-gray-500">
            📊 Graph coming soon...
          </div>
        </div>
      </div>
    </section>
  );
};

export default StudentDashboard;
