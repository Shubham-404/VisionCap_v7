import { useEffect, useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { db } from '../../firebase/config'; // adjust the path if needed
import { doc, getDoc } from 'firebase/firestore';

const StudentProfile = () => {
  const { currentUser } = useAuth();
  const [profileData, setProfileData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchProfile = async () => {
      if (!currentUser) return;

      try {
        const docRef = doc(db, 'students', currentUser.uid);
        const docSnap = await getDoc(docRef);

        if (docSnap.exists()) {
          setProfileData(docSnap.data());
        } else {
          console.error('No such document!');
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
            <p className="text-gray-500 text-sm">Roll No: {profileData.rollNo || 'N/A'}</p>
            <p className="text-gray-500 text-sm">Email: {currentUser.email}</p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="bg-blue-50 p-4 rounded-lg">
            <p className="text-sm text-blue-700 font-medium">Department</p>
            <h4 className="font-semibold text-gray-700">{profileData.department || 'N/A'}</h4>
          </div>
          <div className="bg-green-50 p-4 rounded-lg">
            <p className="text-sm text-green-700 font-medium">Semester</p>
            <h4 className="font-semibold text-gray-700">{profileData.semester || 'N/A'}</h4>
          </div>
        </div>
      </div>
    </section>
  );
};

export default StudentProfile;
