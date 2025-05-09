import { db } from '../config';
import {
  collection,
  doc,
  setDoc,
  getDoc,
  getDocs,
  updateDoc,
  deleteDoc,
  addDoc,
  serverTimestamp,
  // deleteUser,
} from 'firebase/firestore';
import { createUserWithEmailAndPassword, signOut } from 'firebase/auth';
import { auth } from '../config';

// User Profile Operations
export const createUserProfile = async (userId, userData) => {
  try {
    const userRef = doc(db, 'users', userId);
    await setDoc(userRef, {
      ...userData,
      createdAt: serverTimestamp(),
      updatedAt: serverTimestamp(),
    });
    return userRef;
  } catch (error) {
    throw new Error('Error creating user profile: ' + error.message);
  }
};

export const getUserProfile = async (userId) => {
  try {
    const userRef = doc(db, 'users', userId);
    const userSnap = await getDoc(userRef);
    if (userSnap.exists()) {
      return { id: userSnap.id, ...userSnap.data() };
    }
    return null;
  } catch (error) {
    throw new Error('Error fetching user profile: ' + error.message);
  }
};

// Demo Request Operations
export const saveDemoRequest = async (data) => {
  try {
    const demoRequestsRef = collection(db, 'demoRequests');
    const demoDoc = await addDoc(demoRequestsRef, {
      ...data,
      createdAt: serverTimestamp(),
    });
    return demoDoc.id;
  } catch (error) {
    throw new Error('Error saving demo request: ' + error.message);
  }
};

// Organization Services Operations
export const updateOrganizationServices = async (orgId, services) => {
  try {
    const orgRef = doc(db, 'organizations', orgId);
    await setDoc(orgRef, {
      activeServices: services,
      updatedAt: serverTimestamp(),
    }, { merge: true });
  } catch (error) {
    throw new Error('Error updating organization services: ' + error.message);
  }
};

// Camera Operations
export const addCamera = async (orgId, cameraData) => {
  try {
    const camerasRef = collection(db, `organizations/${orgId}/cameras`);
    const cameraDoc = await addDoc(camerasRef, {
      ...cameraData,
      createdAt: serverTimestamp(),
      updatedAt: serverTimestamp(),
    });
    return cameraDoc.id;
  } catch (error) {
    throw new Error('Error adding camera: ' + error.message);
  }
};

export const getCameras = async (orgId) => {
  try {
    const camerasRef = collection(db, `organizations/${orgId}/cameras`);
    const querySnapshot = await getDocs(camerasRef);
    return querySnapshot.docs.map((doc) => ({ id: doc.id, ...doc.data() }));
  } catch (error) {
    throw new Error('Error fetching cameras: ' + error.message);
  }
};

export const updateCamera = async (orgId, cameraId, cameraData) => {
  try {
    const cameraRef = doc(db, `organizations/${orgId}/cameras`, cameraId);
    await updateDoc(cameraRef, {
      ...cameraData,
      updatedAt: serverTimestamp(),
    });
  } catch (error) {
    throw new Error('Error updating camera: ' + error.message);
  }
};

export const deleteCamera = async (orgId, cameraId) => {
  try {
    const cameraRef = doc(db, `organizations/${orgId}/cameras`, cameraId);
    await deleteDoc(cameraRef);
  } catch (error) {
    throw new Error('Error deleting camera: ' + error.message);
  }
};

// Teacher utility
export const addTeacher = async (orgId, teacherData) => {
  let teacherUser = null;
  try {
    // Validate input
    if (!teacherData.email || !teacherData.name) {
      throw new Error('Teacher email and name are required');
    }

    // Store current admin user
    const adminUser = auth.currentUser;
    if (!adminUser) {
      throw new Error('No admin user is currently signed in');
    }

    // Create Firebase Auth user for teacher with retry
    let attempts = 0;
    const maxAttempts = 3;
    while (attempts < maxAttempts) {
      try {
        const userCredential = await createUserWithEmailAndPassword(auth, teacherData.email, '123456');
        teacherUser = userCredential.user;
        break;
      } catch (authError) {
        attempts++;
        if (authError.code === 'auth/email-already-in-use') {
          throw new Error('This email is already registered. Please use a different email or delete the existing user in Firebase Console.');
        } else if (authError.code === 'auth/invalid-email') {
          throw new Error('Invalid email format.');
        } else if (authError.code === 'auth/operation-not-allowed') {
          throw new Error('Email/password authentication is disabled. Please enable it in Firebase Console.');
        } else if (attempts >= maxAttempts) {
          throw new Error(`Failed to create teacher after ${maxAttempts} attempts: ${authError.message}`);
        }
        console.warn(`Retrying createUserWithEmailAndPassword (${attempts}/${maxAttempts}): ${authError.message}`);
        await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1s before retry
      }
    }

    // Sign out to prevent teacher login
    try {
      await signOut(auth);
    } catch (signOutError) {
      console.error('Failed to sign out teacher:', signOutError);
      // Continue with Firestore writes
    }

    // Add to users collection with retry
    let userProfileSuccess = false;
    attempts = 0;
    while (attempts < maxAttempts) {
      try {
        await setDoc(doc(db, 'users', teacherUser.uid), {
          email: teacherData.email,
          displayName: teacherData.name,
          role: 'teacher',
          orgIds: [orgId],
          createdAt: serverTimestamp(),
          updatedAt: serverTimestamp(),
        });
        userProfileSuccess = true;
        break;
      } catch (firestoreError) {
        attempts++;
        if (attempts >= maxAttempts) {
          console.error('Failed to create user profile after retries:', firestoreError);
          throw new Error(`Failed to create user profile: ${firestoreError.message}`);
        }
        console.warn(`Retrying user profile write (${attempts}/${maxAttempts}): ${firestoreError.message}`);
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }

    // Add to organization's users subcollection with retry
    attempts = 0;
    while (attempts < maxAttempts) {
      try {
        await setDoc(doc(db, `organizations/${orgId}/users`, teacherUser.uid), {
          role: 'teacher',
        });
        break;
      } catch (firestoreError) {
        attempts++;
        if (attempts >= maxAttempts) {
          console.error('Failed to add teacher to organization users after retries:', firestoreError);
          throw new Error(`Failed to add teacher to organization users: ${firestoreError.message}`);
        }
        console.warn(`Retrying organization users write (${attempts}/${maxAttempts}): ${firestoreError.message}`);
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }

    // Add to organization's teachers subcollection with retry
    attempts = 0;
    while (attempts < maxAttempts) {
      try {
        await setDoc(doc(db, `organizations/${orgId}/teachers`, teacherUser.uid), {
          name: teacherData.name,
          email: teacherData.email,
          createdAt: serverTimestamp(),
          updatedAt: serverTimestamp(),
        });
        break;
      } catch (firestoreError) {
        attempts++;
        if (attempts >= maxAttempts) {
          console.error('Failed to add teacher to teachers collection after retries:', firestoreError);
          throw new Error(`Failed to add teacher to teachers collection: ${firestoreError.message}`);
        }
        console.warn(`Retrying teachers collection write (${attempts}/${maxAttempts}): ${firestoreError.message}`);
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }

    return { id: teacherUser.uid, ...teacherData };
  } catch (error) {
    console.error('addTeacher error:', error);
    // Cleanup: Delete auth user if Firestore writes failed
    if (teacherUser) {
      try {
        await deleteUser(teacherUser);
        console.log(`Cleaned up orphaned auth user: ${teacherUser.uid}`);
      } catch (deleteError) {
        console.error('Failed to delete orphaned auth user:', deleteError);
      }
    }
    throw error; // Propagate to AdminDashboard.jsx
  }
};