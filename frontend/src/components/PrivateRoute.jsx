import { Navigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { useEffect, useState } from 'react';
import { doc, getDoc } from 'firebase/firestore';
import { db } from '../firebase/config';

export default function PrivateRoute({ children, allowedRoles = [] }) {
  const { currentUser, logout } = useAuth();
  const [userRole, setUserRole] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchUserRole = async () => {
      if (currentUser) {
        try {
          const docRef = doc(db, 'users', currentUser.uid);
          const docSnap = await getDoc(docRef);
          if (docSnap.exists()) {
            const data = docSnap.data();
            setUserRole(data.role);
          } else {
            console.warn("No role found for user.");
            await logout(); // log out user if no role
          }
        } catch (error) {
          console.error("Error fetching role:", error);
        }
      }
      setLoading(false);
    };

    fetchUserRole();
  }, [currentUser, logout]);

  if (!currentUser) {
    alert("Kindly login first.");
    return <Navigate to="/login" />;
  }

  if (loading) return null; // or a spinner

  if (allowedRoles.length && !allowedRoles.includes(userRole)) {
    alert("Access denied. Not authorized for this page.");
    return <Navigate to="/" />;
  }

  return children;
}
