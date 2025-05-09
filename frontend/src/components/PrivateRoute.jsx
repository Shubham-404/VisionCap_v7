import { Navigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

function PrivateRoute({ children, allowedRoles }) {
  const { currentUser, userProfile } = useAuth();

  if (!currentUser) {
    alert("Kindly Login")
    return <Navigate to="/login" />;
  }

  if (!userProfile || !allowedRoles.includes(userProfile.role)) {
    return <Navigate to="/" />;

  }

  return children;
}

export default PrivateRoute;