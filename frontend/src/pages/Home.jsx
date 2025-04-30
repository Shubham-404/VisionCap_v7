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
    <section className="min-h-screen bg-gradient-to-b from-slate-50 to-white text-gray-800 px-6 scroll-smooth">
      {/* Hero */}
      <div className="max-w-6xl mx-auto text-center pt-24 pb-20 animate-fadeIn">
        <h1 className="text-5xl font-bold mb-6 leading-tight">
          AI-Powered Video Analytics for Smarter Spaces
        </h1>
        <p className="text-lg text-gray-600 mb-8 max-w-3xl mx-auto">
          Transform any camera-enabled environment — classrooms, offices, retail, or online sessions — with intelligent behavior tracking and real-time engagement analytics.
        </p>

        {loading ? null : currentUser && role ? (
          <button
            onClick={handleGoToDashboard}
            className="bg-purple-600 hover:bg-purple-700 text-white font-semibold py-3 px-8 rounded-lg shadow-lg transition duration-300 hover:scale-105"
          >
            Go to Dashboard
          </button>
        ) : (
          <div className="flex flex-col sm:flex-row justify-center gap-4">
            <Link
              to="/login"
              className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-8 rounded-lg shadow transition duration-300 hover:scale-105"
            >
              Login
            </Link>
            <Link
              to="mailto:contact@yourcompany.com"
              className="bg-green-600 hover:bg-green-700 text-white font-semibold py-3 px-8 rounded-lg shadow transition duration-300 hover:scale-105"
            >
              Partner With Us
            </Link>
          </div>
        )}
      </div>

      {/* Feature Highlights */}
      <div className="max-w-6xl mx-auto py-12 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8">
        {[
          {
            icon: "🎥",
            title: "Compatible with Any Camera",
            desc: "Easily integrate with standard webcams or CCTV for smart analytics.",
          },
          {
            icon: "📊",
            title: "Behavior & Emotion Detection",
            desc: "Track attention, mood, posture, and anomalies in real-time.",
          },
          {
            icon: "📁",
            title: "Session Summaries",
            desc: "Export attendance, engagement, and event logs with one click.",
          },
          {
            icon: "⚙️",
            title: "Flexible API Integration",
            desc: "Seamlessly plug into your LMS, HR tools, or surveillance systems.",
          },
        ].map(({ icon, title, desc }, idx) => (
          <div
            key={idx}
            className="bg-white p-6 rounded-xl shadow text-center hover:shadow-lg transition duration-300 hover:scale-[1.03]"
          >
            <div className="text-3xl mb-3">{icon}</div>
            <h3 className="text-lg font-semibold">{title}</h3>
            <p className="text-sm text-gray-600 mt-2">{desc}</p>
          </div>
        ))}
      </div>

      {/* Why Us */}
      <div className="bg-slate-100 py-16 animate-fadeIn delay-200">
        <div className="max-w-5xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-4">Why Choose Our Platform?</h2>
          <p className="text-gray-600 mb-10 max-w-2xl mx-auto">
            Our edge-AI technology delivers real-time, privacy-conscious insights from any environment — without the need for invasive hardware or manual supervision.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              { title: "⚡ Industry-Agnostic", desc: "From classrooms to call centers — our tech fits everywhere." },
              { title: "🔐 Privacy Compliant", desc: "Built with GDPR and FERPA standards in mind." },
              { title: "📡 Edge AI Capable", desc: "Run analysis locally with low-latency performance." },
            ].map(({ title, desc }, idx) => (
              <div key={idx} className="p-6 bg-white rounded-xl shadow hover:shadow-md transition">
                <h4 className="text-lg font-semibold mb-2">{title}</h4>
                <p className="text-sm text-gray-600">{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Use Cases */}
      <div className="max-w-6xl mx-auto py-16 px-4 animate-fadeInUp">
        <h2 className="text-3xl font-bold text-center mb-10">Versatile Across Industries</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-8">
          {[
            { title: "🏫 Smart Classrooms", desc: "Automated engagement tracking and attendance." },
            { title: "🏢 Corporate Training", desc: "Measure participation and focus during meetings or e-learning." },
            { title: "🛍️ Retail & Public Spaces", desc: "Analyze customer emotions, footfall, and dwell time patterns." },
          ].map(({ title, desc }, idx) => (
            <div key={idx} className="bg-white p-6 rounded-xl shadow text-center hover:shadow-lg transition">
              <h4 className="text-lg font-semibold mb-2">{title}</h4>
              <p className="text-sm text-gray-600">{desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* CTA */}
      <div className="bg-purple-700 py-12 text-center text-white animate-fadeInUp delay-300">
        <h2 className="text-3xl font-bold mb-4">Empower Your Space with Video Intelligence</h2>
        <p className="text-md mb-6">AI-driven insights that scale with your vision.</p>
        <Link
          to="/login"
          className="bg-white text-purple-700 font-semibold py-2 px-6 rounded-lg shadow hover:bg-gray-100 transition"
        >
          Request a Demo
        </Link>
      </div>

      {/* Trust Banner */}
      <div className="py-12 text-center text-gray-500 text-sm">
        <p>Trusted by forward-thinking educators, enterprises, and developers worldwide.</p>
      </div>
    </section>
  );
};

export default Home;
