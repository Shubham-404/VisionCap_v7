import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { AuthProvider } from './contexts/AuthContext';

import PrivateRoute from "./components/PrivateRoute";
import Layout from "./components/Layout";

import StudentDashboard from "./pages/student/Dashboard";
import StudentProfile from "./pages/student/Profile";
import TeacherDashboard from "./pages/teacher/TeacherDashboard";
import OrgDashboard from "./pages/organization/Dashboard";

import Login from "./pages/Login";
import Home from "./pages/Home";
import Register from "./pages/Register";

function App() {
  return (
    <main className="h-full">
      <AuthProvider>
        <Router>
          <Routes>
            <Route element={<Layout />}>
              <Route path="/" element={<Home />} />

              <Route
                path="/dashboard"
                element={
                  <PrivateRoute allowedRoles={['student']}>
                    <StudentDashboard />
                  </PrivateRoute>
                }
              />
              <Route
                path="/profile"
                element={
                  <PrivateRoute allowedRoles={['student']}>
                    <StudentProfile />
                  </PrivateRoute>
                }
              />
              <Route
                path="/teacher/dashboard"
                element={
                  <PrivateRoute allowedRoles={['teacher']}>
                    <TeacherDashboard />
                  </PrivateRoute>
                }
              />
              <Route
                path="/organizer/dashboard"
                element={
                  <PrivateRoute allowedRoles={['organizer']}>
                    <OrgDashboard />
                  </PrivateRoute>
                }
              />
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
