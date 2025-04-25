import { useEffect, useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { db } from '../../firebase/config'; // adjust the path if needed
import { doc, getDoc, collection, query, where, getDocs } from 'firebase/firestore';

const StudentProfile = () => {
  const { currentUser } = useAuth();
  const [profileData, setProfileData] = useState(null);
  const [attendanceData, setAttendanceData] = useState(null);
  const [behaviourScore, setBehaviourScore] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchProfile = async () => {
      if (!currentUser) return <Navigate to="/login" />;;

      try {
        // Fetch basic profile data from users collection
        const userDocRef = doc(db, 'users', currentUser.uid);
        const userDocSnap = await getDoc(userDocRef);

        if (userDocSnap.exists()) {
          const userData = userDocSnap.data();
          setProfileData(userData);

          const userUsn = userData.Usn;

          // Fetch latest attendance from attendance-logs collection
          const attendanceQuery = query(
            collection(db, 'attendance-logs'),
            where('Usn', '==', userUsn)
          );
          const attendanceSnap = await getDocs(attendanceQuery);
          let totalAttendance = 0;
          let count = 0;

          attendanceSnap.forEach((doc) => {
            const data = doc.data();
            const students = data.students; // students is an object

            // If multiple students: loop through them
            for (const studentId in students) {
              const student = students[studentId];
              if (student.usn === userUsn && student.present === "true") {
                if (data.attendancePercentage !== undefined) {
                  totalAttendance += data.attendancePercentage;
                  count += 1;
                }
              }
            }
            // });

            if (data.attendancePercentage !== undefined) {
              totalAttendance += data.attendancePercentage;
              count += 1;
            }
          });

          if (count > 0) {
            setAttendanceData((totalAttendance / count).toFixed(2) + '%');
          } else {
            setAttendanceData('No data');
          }

          // Fetch behaviour score from engagement-reports collection
          const behaviourQuery = query(
            collection(db, 'engagement-reports'),
            where('Usn', '==', userUsn)
          );
          const behaviourSnap = await getDocs(behaviourQuery);

          let totalScore = 0;
          let sessionsCount = 0;

          behaviourSnap.forEach((doc) => {
            const data = doc.data();
            if (data.behaviourScore !== undefined) {
              totalScore += data.behaviourScore;
              sessionsCount += 1;
            }
          });

          if (sessionsCount > 0) {
            setBehaviourScore((totalScore / sessionsCount).toFixed(2));
          } else {
            setBehaviourScore('No data');
          }

        } else {
          console.error('No such user document!');
        }
      } catch (error) {
        console.error('Error fetching profile:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchProfile();
  }, [currentUser]);

  if (loading) {
    return (
      <section className="flex justify-center items-center min-h-screen">
        <div className="text-xl font-semibold text-gray-600 animate-pulse">
          Loading profile...
        </div>
      </section>
    );
  }

  if (!profileData) {
    return (
      <section className="flex justify-center items-center min-h-screen">
        <div className="text-xl font-semibold text-red-500">
          Failed to load profile.
        </div>
      </section>
    );
  }

  return (
    <section className="container mx-auto px-4 py-6">
      <h2 className="text-2xl font-semibold text-gray-800 mb-4">My Profile</h2>

      <div className="bg-white shadow rounded-xl p-6 space-y-4">
        <div className="flex items-center space-x-4">
          <img
            src={profileData.photoURL || 'https://via.placeholder.com/80'}
            alt="Profile"
            className="w-20 h-20 rounded-full border-2 border-blue-500 object-cover"
          />
          <div>
            <h3 className="text-xl font-semibold text-gray-700">
              {profileData.name || currentUser.displayName || 'Unnamed User'}
            </h3>
            <p className="text-gray-500 text-sm">USN: {profileData.Usn || 'N/A'}</p>
            <p className="text-gray-500 text-sm">Email: {currentUser.email}</p>
            <p className="text-gray-500 text-sm">Department: {profileData.dept || 'N/A'}</p>
            <p className="text-gray-500 text-sm">Semester: {profileData.sem || 'N/A'}</p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="bg-blue-50 p-4 rounded-lg">
            <p className="text-sm text-blue-700 font-medium">Attendance</p>
            <h4 className="font-semibold text-gray-700">{attendanceData}</h4>
          </div>
          <div className="bg-green-50 p-4 rounded-lg">
            <p className="text-sm text-green-700 font-medium">Behaviour Score</p>
            <h4 className="font-semibold text-gray-700">{behaviourScore}</h4>
          </div>
        </div>
      </div>
    </section>
  );
};

export default StudentProfile;