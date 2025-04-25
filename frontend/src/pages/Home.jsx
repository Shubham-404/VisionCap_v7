import { Link, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import { useAuth } from "../contexts/AuthContext";
import { doc, getDoc } from "firebase/firestore";
import { db } from "../firebase/config";

const Home = () => {
  const { currentUser } = useAuth();
  const navigate = useNavigate();
  const [role, setRole] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchRole = async () => {
      if (currentUser) {
        try {
          const userDoc = await getDoc(doc(db, "users", currentUser.uid));
          if (userDoc.exists()) {
            setRole(userDoc.data().role);
          }
        } catch (err) {
          console.error("Error fetching user role:", err);
        }
      }
      setLoading(false);
    };

    fetchRole();
  }, [currentUser]);

  const handleGoToDashboard = () => {
    if (role === "student") navigate("/dashboard");
    else if (role === "teacher") navigate("/teacher/dashboard");
  };

  return (
    <section className="min-h-screen flex flex-col justify-center items-center bg-gradient-to-br from-slate-100 to-slate-200 px-4">
      <div className="max-w-3xl text-center">
        <h1 className="text-4xl md:text-5xl font-bold text-gray-800 mb-4">
          Welcome to the Student Engagement Platform 🎓
        </h1>
        <p className="text-lg text-gray-600 mb-6">
          Track your attendance, monitor your behavior score, and stay informed – all in one place.
        </p>

        {loading ? null : currentUser && role ? (
          <button
            onClick={handleGoToDashboard}
            className="bg-purple-600 hover:bg-purple-700 text-white font-semibold py-2 px-6 rounded-lg shadow transition"
          >
            Go to Dashboard
          </button>
        ) : (
          <div className="flex flex-col sm:flex-row justify-center gap-4">
            <Link
              to="/login"
              className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg shadow transition"
            >
              Student Login
            </Link>
            <Link
              to="/signup"
              className="bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-6 rounded-lg shadow transition"
            >
              New Student? Sign Up
            </Link>
          </div>
        )}
      </div>

      <div className="mt-16 grid grid-cols-1 sm:grid-cols-3 gap-6 max-w-5xl w-full">
        <div className="bg-white shadow rounded-xl p-6 text-center">
          <h3 className="text-lg font-semibold text-gray-700">📅 Attendance Logs</h3>
          <p className="text-sm text-gray-500 mt-2">View your attendance across sessions with clarity.</p>
        </div>
        <div className="bg-white shadow rounded-xl p-6 text-center">
          <h3 className="text-lg font-semibold text-gray-700">📈 Engagement Reports</h3>
          <p className="text-sm text-gray-500 mt-2">Track your behavior and engagement score over time.</p>
        </div>
        <div className="bg-white shadow rounded-xl p-6 text-center">
          <h3 className="text-lg font-semibold text-gray-700">🧠 Smart Insights</h3>
          <p className="text-sm text-gray-500 mt-2">Understand trends and get suggestions to improve.</p>
        </div>
      </div>
    </section>
  );
};

export default Home;
