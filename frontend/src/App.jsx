import { useEffect } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { AuthProvider } from './contexts/AuthContext';
import PrivateRoute from "./components/PrivateRoute";
import Layout from "./components/Layout";
import StudentDashboard from "./pages/student/Dashboard";
import StudentProfile from "./pages/student/Profile";
import AdminDashboard from "./pages/teacher/TeacherDashboard";
import Login from "./pages/Login";
import Register from "./pages/Register";

import TeacherDashboard from "./pages/teacher/TeacherDashboard";
import OrgDashboard from "./pages/organization/Dashboard";

function App() {
  return (
    <main className="h-full">

      <AuthProvider>
        <Router>
          <Routes>
            <Route element={<Layout />}>
              <Route path="/" element={
                <PrivateRoute>
                  <StudentDashboard />
                </PrivateRoute>
              } />
              <Route path="/profile" element={
                <PrivateRoute>
                  <StudentProfile />
                </PrivateRoute>
              } />

              <Route path="/admin" element={
                <PrivateRoute>
                  <AdminDashboard />
                </PrivateRoute>} />

              <Route path="/login" element={<Login />} />

              <Route path="/register" element={<Register />} />

            </Route>
          </Routes>
        </Router>
      </AuthProvider>
    </main>

  );
}

export default App;
