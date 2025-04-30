
import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate, Link } from 'react-router-dom';
import { doc, getDoc } from 'firebase/firestore';
import { db } from '../firebase/config';

const Login = () => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const { login, logout } = useAuth();
    const navigate = useNavigate();

    const handleLogin = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);

        try {
            const userCredential = await login(email, password);
            const user = userCredential.user;

            const docRef = doc(db, 'users', user.uid);
            const docSnap = await getDoc(docRef);

            if (docSnap.exists()) {
                const userData = docSnap.data();
                const role = userData.role;

                if (!role) {
                    await logout();
                    setError('No role assigned. Please contact the admin.');
                    return;
                }

                if (role === 'student') {
                    navigate('/');
                } else if (role === 'teacher') {
                    navigate('/teacher/dashboard');
                } else if (role === 'organizer') {
                    navigate('/organizer/dashboard');
                } else {
                    await logout();
                    setError('Unknown role. Please contact admin.');
                }
            } else {
                await logout();
                setError('No user data found in Firestore.');
            }
        } catch (error) {
            setError('Failed to sign in: ' + error.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <section className="min-h-[80vh] h-full flex items-center justify-center bg-gradient-to-br from-white to-blue-500 bg-no-repeat bg-bottom bg-cover">
            <div className="bg-white/90 backdrop-blur-xs w-full max-w-md p-8 rounded-xl shadow-lg">
                <h2 className="text-3xl font-bold text-center text-blue-700 mb-6">Sign In</h2>
                {error && (
                    <div className="bg-red-100 text-red-700 px-4 py-2 rounded mb-4 text-sm">
                        {error}
                    </div>
                )}

                <form onSubmit={handleLogin} className="space-y-4">
                    <div>
                        <input
                            type="email"
                            className="w-full px-4 py-2 border border-gray-300 rounded-lg outline-blue-400"
                            placeholder="Email/USN"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                        />
                    </div>

                    <div>
                        <input
                            type="password"
                            className="w-full px-4 py-2 border border-gray-300 rounded-lg outline-blue-400"
                            placeholder="Password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                        />
                    </div>

                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition"
                    >
                        {loading ? 'Logging in...' : 'Login'}
                    </button>
                </form>

                <p className="text-sm text-center mt-6">
                    Not Registered?{' '}
                    <Link to="mailto:organizeremail@org.ac.in" className="text-blue-600 hover:underline">
                        Ask here
                    </Link>
                </p>
            </div>
        </section>
    );
};

export default Login;
