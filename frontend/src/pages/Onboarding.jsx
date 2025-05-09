import { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import { doc, setDoc, collection, addDoc, serverTimestamp } from 'firebase/firestore';
import { db } from '../firebase/config';

function Onboarding() {
  const [orgName, setOrgName] = useState('');
  const [role, setRole] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { currentUser } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (!currentUser) {
      setError('Please log in to complete onboarding.');
      navigate('/login');
    }
  }, [currentUser, navigate]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    if (!currentUser) {
      setError('No user logged in. Please log in again.');
      setLoading(false);
      return;
    }

    // Validate orgName for organization role
    if (role === 'organization' && !orgName.trim()) {
      setError('Organization Name is required for Organization role.');
      setLoading(false);
      return;
    }

    try {
      // Create organization only for organization role
      let orgId;
      if (role === 'organization') {
        const orgRef = await addDoc(collection(db, 'organizations'), {
          name: orgName.trim(),
          activeServices: [],
          createdAt: serverTimestamp(),
          updatedAt: serverTimestamp(),
        });
        orgId = orgRef.id;

        // Add user to organization's users subcollection
        await setDoc(doc(db, `organizations/${orgId}/users`, currentUser.uid), {
          role,
        });
      }

      // Save user data
      await setDoc(doc(db, 'users', currentUser.uid), {
        email: currentUser.email,
        displayName: currentUser.displayName || currentUser.email.split('@')[0],
        role,
        orgIds: orgId ? [orgId] : [],
        createdAt: serverTimestamp(),
        updatedAt: serverTimestamp(),
      });

      // Navigate based on role
      if (role === 'organization') {
        navigate('/admin/dashboard');
      } else if (role === 'teacher') {
        navigate('/teacher/dashboard');
      } else if (role === 'student') {
        navigate('/dashboard');
      }
    } catch (err) {
      setError('Failed to complete onboarding: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="min-h-[80vh] h-full flex items-center justify-center bg-gradient-to-br from-white to-blue-500 bg-no-repeat bg-bottom bg-cover">
      <div className="bg-white/90 backdrop-blur-xs w-full max-w-md p-8 rounded-xl shadow-lg">
        <h2 className="text-3xl font-bold text-center text-blue-700 mb-6">Complete Your Setup</h2>
        {error && (
          <div className="bg-red-100 text-red-700 px-4 py-2 rounded mb-4 text-sm">
            {error}
          </div>
        )}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <input
              type="text"
              value={orgName}
              onChange={(e) => setOrgName(e.target.value)}
              placeholder="Organization Name (required only for Organization role)"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg outline-blue-400"
            />
          </div>
          <div>
            <select
              value={role}
              onChange={(e) => setRole(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg outline-blue-400"
              required
            >
              <option value="">Select Role</option>
              <option value="student">Student</option>
              <option value="teacher">Teacher</option>
              <option value="organization">Organization</option>
            </select>
          </div>
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition"
          >
            {loading ? 'Saving...' : 'Complete Setup'}
          </button>
        </form>
      </div>
    </section>
  );
}

export default Onboarding;