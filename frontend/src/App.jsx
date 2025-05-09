import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { useEffect, useState, Component } from 'react';

import { AuthProvider } from './contexts/AuthContext';
import PrivateRoute from './components/PrivateRoute';
import Layout from './components/Layout';
import Loader from './components/Loader';

import StudentDashboard from './pages/student/Dashboard';
import StudentProfile from './pages/student/Profile';
import TeacherDashboard from './pages/teacher/TeacherDashboard';
import AdminDashboard from './pages/organization/Dashboard';
import Home from './pages/Home';
import Login from './pages/Login';
import Signup from './pages/Signup';
import Onboarding from './pages/Onboarding';
import NotFound from './pages/NotFound';
import Services from './pages/Services';
import About from './pages/About';
import Contact from './pages/Contact';

function AppRoutesWithLoader() {

  class ErrorBoundary extends Component {
    state = { hasError: false };
    static getDerivedStateFromError() {
      return { hasError: true };
    }
    render() {
      if (this.state.hasError) {
        return <h1>Something went wrong. Please try again.</h1>;
      }
      return this.props.children;
    }
  }


  const location = useLocation();
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    const timeout = setTimeout(() => {
      setLoading(false);

    }, 500);

    return () => clearTimeout(timeout);
  }, [location]);

  return (
    <>
      {loading && <Loader />}
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Home />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/onboarding" element={<Onboarding />} />
          <Route path="/about" element={<About />} />
          <Route path="/contact" element={<Contact />} />

          <Route path="/services" element={
            <ErrorBoundary>
              <Services />
            </ErrorBoundary>
          } />
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
            path="/admin/dashboard"
            element={
              <PrivateRoute allowedRoles={['organization']}>
                <AdminDashboard />
              </PrivateRoute>
            }
          />
        
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </>
  );
}

function App() {
  return (
    <main className="h-full">
      <AuthProvider>
        <Router>
          <AppRoutesWithLoader />
        </Router>
      </AuthProvider>
    </main>
  );
}

export default App;