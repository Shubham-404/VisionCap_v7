import { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { doc, getDoc, collection, getDocs } from 'firebase/firestore';
import { db } from '../../firebase/config';
import { addCamera, getCameras, updateCamera, deleteCamera, addTeacher } from '../../firebase/utils/firestore';
import { useNavigate } from 'react-router-dom';

function AdminDashboard() {
  const { userProfile } = useAuth();
  const navigate = useNavigate();
  const [selectedModules, setSelectedModules] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [cameras, setCameras] = useState([]);
  const [cameraForm, setCameraForm] = useState({ name: '', streamUrl: '', id: null });
  const [cameraError, setCameraError] = useState('');
  const [cameraSuccess, setCameraSuccess] = useState('');
  const [teacherForm, setTeacherForm] = useState({ name: '', email: '' });
  const [teacherError, setTeacherError] = useState('');
  const [teacherSuccess, setTeacherSuccess] = useState('');
  const [teachers, setTeachers] = useState([]);
  const [teacherLoading, setTeacherLoading] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      if (!userProfile || !userProfile.orgIds?.[0]) {
        setError('No organization assigned.');
        setLoading(false);
        return;
      }

      try {
        // Fetch services
        const orgDoc = await getDoc(doc(db, 'organizations', userProfile.orgIds[0]));
        if (orgDoc.exists()) {
          setSelectedModules(orgDoc.data().activeServices || []);
        } else {
          setError('Organization not found.');
        }
        // Fetch cameras
        const cameraList = await getCameras(userProfile.orgIds[0]);
        setCameras(cameraList);
        // Fetch teachers
        const teachersSnapshot = await getDocs(collection(db, `organizations/${userProfile.orgIds[0]}/teachers`));
        const teacherList = teachersSnapshot.docs.map(doc => ({ id: doc.id, ...doc.data() }));
        setTeachers(teacherList);
        console.log('Teachers fetched:', teacherList);
      } catch (err) {
        setError('Failed to load data: ' + err.message);
      }
      setLoading(false);
    };
    fetchData();
  }, [userProfile]);

  const handleCameraFormChange = (e) => {
    const { name, value } = e.target;
    setCameraForm((prev) => ({ ...prev, [name]: value }));
  };

  const handleCameraSubmit = async (e) => {
    e.preventDefault();
    setCameraError('');
    setCameraSuccess('');
    if (!cameraForm.name || !cameraForm.streamUrl) {
      setCameraError('Camera name and stream URL are required.');
      return;
    }
    try {
      if (cameraForm.id) {
        await updateCamera(userProfile.orgIds[0], cameraForm.id, {
          name: cameraForm.name,
          streamUrl: cameraForm.streamUrl,
        });
        setCameraSuccess('Camera updated successfully!');
      } else {
        await addCamera(userProfile.orgIds[0], {
          name: cameraForm.name,
          streamUrl: cameraForm.streamUrl,
        });
        setCameraSuccess('Camera added successfully!');
      }
      const cameraList = await getCameras(userProfile.orgIds[0]);
      setCameras(cameraList);
      setCameraForm({ name: '', streamUrl: '', id: null });
    } catch (err) {
      setCameraError('Failed to save camera: ' + err.message);
    }
  };

  const handleEditCamera = (camera) => {
    setCameraForm({ name: camera.name, streamUrl: camera.streamUrl, id: camera.id });
    setCameraError('');
    setCameraSuccess('');
  };

  const handleDeleteCamera = async (cameraId) => {
    if (window.confirm('Are you sure you want to delete this camera?')) {
      try {
        await deleteCamera(userProfile.orgIds[0], cameraId);
        const cameraList = await getCameras(userProfile.orgIds[0]);
        setCameras(cameraList);
        setCameraSuccess('Camera deleted successfully!');
      } catch (err) {
        setCameraError('Failed to delete camera: ' + err.message);
      }
    }
  };

  const handleTeacherFormChange = (e) => {
    const { name, value } = e.target;
    setTeacherForm((prev) => ({ ...prev, [name]: value }));
  };

  const handleTeacherSubmit = async (e) => {
    e.preventDefault();
    setTeacherError('');
    setTeacherSuccess('');
    setTeacherLoading(true);

    if (!teacherForm.name || !teacherForm.email) {
      setTeacherError('Teacher name and email are required.');
      setTeacherLoading(false);
      return;
    }

    try {
      console.log('Adding teacher:', teacherForm);
      console.log('User profile:', userProfile);
      await addTeacher(userProfile.orgIds[0], {
        name: teacherForm.name,
        email: teacherForm.email,
      });
      setTeacherSuccess('Teacher added successfully! They can log in with email and password: 123456. Please sign in again.');
      const teachersSnapshot = await getDocs(collection(db, `organizations/${userProfile.orgIds[0]}/teachers`));
      const teacherList = teachersSnapshot.docs.map(doc => ({ id: doc.id, ...doc.data() }));
      setTeachers(teacherList);
      console.log('Teachers updated:', teacherList);
      setTeacherForm({ name: '', email: '' });
    } catch (err) {
      console.error('Teacher error:', err);
      setTeacherError(`Failed to add teacher: ${err.message}`);
    } finally {
      setTeacherLoading(false);
    }
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
        <h1 className="text-3xl font-bold mb-4">Admin Dashboard</h1>
        <p className="text-lg text-gray-600 mb-8">
          Manage your organization’s settings, view active AI-powered video analytics services, and add teachers.
        </p>
        {error && (
          <div className="bg-red-100 text-red-700 px-4 py-2 rounded mb-4 text-sm">{error}</div>
        )}

        <div className="bg-white p-6 rounded-xl shadow hover:shadow-lg transition duration-300 mb-12">
          <h2 className="text-2xl font-semibold mb-4">Active Services</h2>
          {selectedModules.length > 0 ? (
            <div className="flex flex-wrap gap-2">
              {selectedModules.map((service) => (
                <span
                  key={service}
                  className="inline-block bg-blue-100 text-blue-800 text-sm font-semibold px-3 py-1 rounded-full"
                >
                  {service}
                </span>
              ))}
            </div>
          ) : (
            <p className="text-sm text-gray-600">
              No services selected.{' '}
              <button
                onClick={() => navigate('/services')}
                className="text-blue-600 hover:underline"
              >
                Go to Services page to select.
              </button>
            </p>
          )}
          <button
            onClick={() => navigate('/services')}
            className="mt-4 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg shadow transition duration-300 hover:scale-105"
          >
            Edit Services
          </button>
        </div>

        <div className="bg-white p-6 rounded-xl shadow hover:shadow-lg transition duration-300 mb-12">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Teacher Management</h3>
          <form onSubmit={handleTeacherSubmit} className="mb-6">
            {teacherError && (
              <div className="bg-red-100 text-red-700 px-4 py-2 rounded mb-4 text-sm">{teacherError}</div>
            )}
            {teacherSuccess && (
              <div className="bg-green-100 text-green-700 px-4 py-2 rounded mb-4 text-sm">{teacherSuccess}</div>
            )}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-1">Teacher Name</label>
                <input
                  type="text"
                  name="name"
                  value={teacherForm.name}
                  onChange={handleTeacherFormChange}
                  className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
                  required
                  disabled={teacherLoading}
                />
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-1">Teacher Email</label>
                <input
                  type="email"
                  name="email"
                  value={teacherForm.email}
                  onChange={handleTeacherFormChange}
                  className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
                  required
                  disabled={teacherLoading}
                />
              </div>
            </div>
            <button
              type="submit"
              disabled={teacherLoading}
              className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg shadow transition duration-300 hover:scale-105 disabled:opacity-50"
            >
              {teacherLoading ? 'Adding Teacher...' : 'Add Teacher'}
            </button>
          </form>

          {teachers.length > 0 ? (
            <div className="bg-gray-50 p-4 rounded-lg">
              <h4 className="text-md font-semibold text-gray-800 mb-2">Registered Teachers</h4>
              <ul className="space-y-2">
                {teachers.map((teacher) => (
                  <li key={teacher.id} className="flex justify-between items-center p-2 bg-white rounded-lg shadow">
                    <div>
                      <span className="font-semibold">{teacher.name}</span> - {teacher.email}
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <p className="text-sm text-gray-600">No teachers registered yet.</p>
          )}
        </div>

        <div className="bg-white p-6 rounded-xl shadow hover:shadow-lg transition duration-300 mb-12">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Camera Management</h3>
          <form onSubmit={handleCameraSubmit} className="mb-6">
            {cameraError && (
              <div className="bg-red-100 text-red-700 px-4 py-2 rounded mb-4 text-sm">{cameraError}</div>
            )}
            {cameraSuccess && (
              <div className="bg-green-100 text-green-700 px-4 py-2 rounded mb-4 text-sm">{cameraSuccess}</div>
            )}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-1">Camera Name</label>
                <input
                  type="text"
                  name="name"
                  value={cameraForm.name}
                  onChange={handleCameraFormChange}
                  className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-1">Stream URL</label>
                <input
                  type="url"
                  name="streamUrl"
                  value={cameraForm.streamUrl}
                  onChange={handleCameraFormChange}
                  className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
                  required
                />
              </div>
            </div>
            <button
              type="submit"
              className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg shadow transition duration-300 hover:scale-105"
            >
              {cameraForm.id ? 'Update Camera' : 'Add Camera'}
            </button>
            {cameraForm.id && (
              <button
                type="button"
                onClick={() => setCameraForm({ name: '', streamUrl: '', id: null })}
                className="ml-4 bg-gray-600 hover:bg-gray-700 text-white font-semibold py-2 px-6 rounded-lg shadow transition duration-300 hover:scale-105"
              >
                Cancel
              </button>
            )}
          </form>

          {cameras.length > 0 ? (
            <div className="bg-gray-50 p-4 rounded-lg">
              <h4 className="text-md font-semibold text-gray-800 mb-2">Registered Cameras</h4>
              <ul className="space-y-2">
                {cameras.map((camera) => (
                  <li key={camera.id} className="flex justify-between items-center p-2 bg-white rounded-lg shadow">
                    <div>
                      <span className="font-semibold">{camera.name}</span> - {camera.streamUrl}
                    </div>
                    <div>
                      <button
                        onClick={() => handleEditCamera(camera)}
                        className="text-blue-600 hover:text-blue-800 mr-4"
                      >
                        Edit
                      </button>
                      <button
                        onClick={() => handleDeleteCamera(camera.id)}
                        className="text-red-600 hover:text-red-800"
                      >
                        Delete
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <p className="text-sm text-gray-600">No cameras registered yet.</p>
          )}
        </div>

        <div className="bg-white p-6 rounded-xl shadow hover:shadow-lg transition duration-300">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Organization Settings</h3>
          <p className="text-sm text-gray-600 mb-4">Configure pricing plans and user roles.</p>
          <p className="text-sm text-gray-600">User role management coming soon.</p>
        </div>
      </div>
    </section>
  );
}

export default AdminDashboard;