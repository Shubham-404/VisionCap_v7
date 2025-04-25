import { Navigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

export default function PrivateRoute({ children }) {
  const { currentUser } = useAuth();
  const notLogged = ()=>{
    alert("Kindly login first.")
    return <Navigate to="/login" />;
  }

  return currentUser ? children : notLogged();
} 