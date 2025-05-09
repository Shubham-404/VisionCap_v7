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

  async function signup(email, password) {
    try {
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      await createUserProfile(userCredential.user.uid, {
        email: userCredential.user.email,
        role: 'student',
        displayName: userCredential.user.displayName || email.split('@')[0],
        photoURL: userCredential.user.photoURL || null,
      });
      return userCredential;
    } catch (error) {
      throw error;
    }
  }

  function login(email, password) {
    return signInWithEmailAndPassword(auth, email, password);
  }

  async function logout() {
    setUserProfile(null);
    try {
      await signOut(auth);
    } catch (error) {
      console.error('Logout failed:', error);
      throw error;
    }
  }

  async function signInWithGoogle() {
    try {
      const provider = new GoogleAuthProvider();
      const userCredential = await signInWithPopup(auth, provider);
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

  function resetPassword(email) {
    return sendPasswordResetEmail(auth, email);
  }

  async function loadUserProfile(user) {
    if (user) {
      try {
        const profile = await getUserProfile(user.uid);
        setUserProfile(profile || null);
        if (!profile) {
          console.warn(`No user profile found for UID: ${user.uid}`);
        }
      } catch (error) {
        console.error('Error loading user profile:', error);
        setUserProfile(null);
      }
    } else {
      setUserProfile(null);
    }
  }

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      setLoading(true);
      setCurrentUser(user);
      await loadUserProfile(user);
      if (!user && window.location.pathname !== '/login' && window.location.pathname !== '/signup') {
        window.location.href = '/login';
      }
      setLoading(false);
    });

    return () => unsubscribe();
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