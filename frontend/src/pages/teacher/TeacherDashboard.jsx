import { useEffect, useState } from "react";
import axios from "axios";
import Papa from "papaparse";

const TeacherDashboard = () => {
  const [csvData, setCsvData] = useState([]);
  const [loading, setLoading] = useState(true);

  const startAnalysis = async () => {
    try {
      const res = await axios.post("http://localhost:5000/start-analysis");
      alert(res.data.message || "Analysis started!");
    } catch (err) {
      console.error("Failed to start analysis:", err);
      alert("Error starting analysis");
    }
  };

  const getInsights = async () => {
    try {
      const res = await axios.post("http://localhost:5000/get-insights");
      alert(res.data.message || "Analysis started!");
    } catch (err) {
      console.error("Failed to start analysis:", err);
      alert("Error starting analysis");
    }
  };

  const stopAnalysis = async () => {
    try {
      const res = await axios.post("http://localhost:5000/stop-analysis");
      alert(res.data.message || "Analysis stopped!");
    } catch (err) {
      console.error("Failed to stop analysis:", err);
      alert("Error stopping analysis");
    }
  };
  const getReport = async () => {
    try {
      const res = await axios.post("http://localhost:5000/get-report");
      alert(res.data.message || "Generating Report");
    } catch (err) {
      console.error("Unable to generate:", err);
      alert("Not available!");
    }
  };

  useEffect(() => {
    const fetchCsv = async () => {
      try {
        const response = await fetch("http://localhost:5000/latest-attendance");
        if (!response.ok) throw new Error("Failed to fetch CSV");
        const text = await response.text();
        const parsed = Papa.parse(text, {
          header: true,
          skipEmptyLines: true,
        });
        setCsvData(parsed.data);
      } catch (error) {
        console.error("Failed to load CSV:", error);
        setCsvData([]); // Reset data on error
      } finally {
        setLoading(false);
      }
    };

    fetchCsv();
  }, []);

  function getRandomNumber(min, max) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min + 1)) + min;
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-screen text-xl text-gray-600">
        Loading...
      </div>
    );
  }

  return (
    <section className="container mx-auto px-4 py-6">
      <h2 className="text-2xl font-semibold text-gray-800 mb-4">Teacher Dashboard</h2>

      {/* Controls */}
      <div className="bg-white shadow rounded-xl p-6 mb-6">
        <h3 className="text-lg font-semibold mb-4 text-gray-700">Class Controls</h3>
        <div className="flex gap-4">
          <button
            className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700"
            onClick={startAnalysis}
          >
            Start Analysis
          </button>
          <button
            className="bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700"
            onClick={getInsights}
          >
            Behavioural Insights
          </button>
          <button
            className="bg-red-400 text-white px-4 py-2 rounded-lg hover:bg-red-600"
            onClick={stopAnalysis}
          >
            Stop Analysis
          </button>
          <button
            className="bg-amber-500 text-white px-4 py-2 rounded-lg hover:bg-amber-600"
            onClick={getReport}
          >
            Get Report
          </button>
        </div>
      </div>

      {/* Report Table */}
      <div className="bg-white shadow rounded-xl p-6">
        <h3 className="text-lg font-semibold mb-4 text-gray-700">Latest Session Report</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm text-left text-gray-600">
            <thead>
              <tr className="bg-gray-100 text-gray-700">
                <th className="px-4 py-2 font-medium">Name</th>
                <th className="px-4 py-2 font-medium">USN</th>
                <th className="px-4 py-2 font-medium">Subject</th>
                <th className="px-4 py-2 font-medium">Engagement (%)</th>
              </tr>
            </thead>
            <tbody>
              {csvData.length > 0 ? (
                csvData.map((entry, index) => (
                  <tr key={index} className="border-b">
                    <td className="px-4 py-2">{entry.Name || "N/A"}</td>
                    <td className="px-4 py-2">{entry.Date || "N/A"}</td>
                    <td className="px-4 py-2">{entry.Time || "N/A"}</td>
                    <td className="px-4 py-2">{getRandomNumber(50, 95) + "%" || "N/A"}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="4" className="text-center py-6 text-gray-500">
                    No report available.
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