import { useEffect } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import StudentDashboard from "./pages/student/Dashboard";
import TeacherDashboard from "./pages/teacher/Dashboard";
import OrgDashboard from "./pages/organization/Dashboard";

function App() {
  useEffect(() => { fetch('/api/ping').then(res => res.json()).then(data => console.log(data)); }, []);
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/student" element={<StudentDashboard />} />
        <Route path="/teacher" element={<TeacherDashboard />} />
        <Route path="/org" element={<OrgDashboard />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
