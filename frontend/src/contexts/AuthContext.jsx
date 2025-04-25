import { createContext, useContext, useState, useEffect } from 'react';
import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  onAuthStateChanged,
  GoogleAuthProvider,
  signInWithPopup,
  sendPasswordResetEmail,
} from 'firebase/auth';
import { auth } from '../firebase/config';
import { createUserProfile, getUserProfile } from '../firebase/utils/firestore';

const AuthContext = createContext();

export function useAuth() {
  return useContext(AuthContext);
}

export function AuthProvider({ children }) {
  const [currentUser, setCurrentUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [userProfile, setUserProfile] = useState(null);

  // Sign up function
  async function signup(email, password) {
    try {
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      // Create user profile in Firestore
      await createUserProfile(userCredential.user.uid, {
        email: userCredential.user.email,
        role: 'student', // Default role
        displayName: userCredential.user.displayName || email.split('@')[0],
        photoURL: userCredential.user.photoURL || null,
      });
      return userCredential;
    } catch (error) {
      throw error;
    }
  }

  // Login function
  function login(email, password) {
    return signInWithEmailAndPassword(auth, email, password);
  }

  // Logout function
  async function logout() {
    setUserProfile(null);
    return signOut(auth);
  }

  // Google Sign in
  async function signInWithGoogle() {
    try {
      const provider = new GoogleAuthProvider();
      const userCredential = await signInWithPopup(auth, provider);
      
      // Check if user profile exists, if not create one
      const profile = await getUserProfile(userCredential.user.uid);
      if (!profile) {
        await createUserProfile(userCredential.user.uid, {
          email: userCredential.user.email,
          role: 'student',
          displayName: userCredential.user.displayName,
          photoURL: userCredential.user.photoURL,
        });
      }
      return userCredential;
    } catch (error) {
      throw error;
    }
  }

  // Reset Password
  function resetPassword(email) {
    return sendPasswordResetEmail(auth, email);
  }

  // Load user profile
  async function loadUserProfile(user) {
    if (user) {
      const profile = await getUserProfile(user.uid);
      setUserProfile(profile);
    } else {
      setUserProfile(null);
    }
  }

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      setCurrentUser(user);
      await loadUserProfile(user);
      setLoading(false);
    });

    return unsubscribe;
  }, []);

  const value = {
    currentUser,
    userProfile,
    signup,
    login,
    logout,
    signInWithGoogle,
    resetPassword,
  };

  return (
    <AuthContext.Provider value={value}>
      {!loading && children}
    </AuthContext.Provider>
  );
} 