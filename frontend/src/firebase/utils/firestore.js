import { db } from '../config';
import {
  collection,
  doc,
  setDoc,
  getDoc,
  getDocs,
  updateDoc,
  deleteDoc,
  query,
  where,
  addDoc,
  serverTimestamp,
} from 'firebase/firestore';

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

// Event Operations
export const createEvent = async (eventData) => {
  try {
    const eventsRef = collection(db, 'events');
    const eventDoc = await addDoc(eventsRef, {
      ...eventData,
      createdAt: serverTimestamp(),
      updatedAt: serverTimestamp(),
    });
    return eventDoc.id;
  } catch (error) {
    throw new Error('Error creating event: ' + error.message);
  }
};

export const getEvent = async (eventId) => {
  try {
    const eventRef = doc(db, 'events', eventId);
    const eventSnap = await getDoc(eventRef);
    if (eventSnap.exists()) {
      return { id: eventSnap.id, ...eventSnap.data() };
    }
    return null;
  } catch (error) {
    throw new Error('Error fetching event: ' + error.message);
  }
};

export const updateEvent = async (eventId, eventData) => {
  try {
    const eventRef = doc(db, 'events', eventId);
    await updateDoc(eventRef, {
      ...eventData,
      updatedAt: serverTimestamp(),
    });
  } catch (error) {
    throw new Error('Error updating event: ' + error.message);
  }
};

export const deleteEvent = async (eventId) => {
  try {
    const eventRef = doc(db, 'events', eventId);
    await deleteDoc(eventRef);
  } catch (error) {
    throw new Error('Error deleting event: ' + error.message);
  }
};

export const getEventsByOrganizer = async (organizerId) => {
  try {
    const eventsRef = collection(db, 'events');
    const q = query(eventsRef, where('organizerId', '==', organizerId));
    const querySnapshot = await getDocs(q);
    return querySnapshot.docs.map(doc => ({
      id: doc.id,
      ...doc.data()
    }));
  } catch (error) {
    throw new Error('Error fetching organizer events: ' + error.message);
  }
};

// Registration Operations
export const registerForEvent = async (eventId, userId, registrationData) => {
  try {
    const registrationRef = collection(db, 'events', eventId, 'registrations');
    await setDoc(doc(registrationRef, userId), {
      ...registrationData,
      userId,
      status: 'pending',
      registeredAt: serverTimestamp(),
    });
  } catch (error) {
    throw new Error('Error registering for event: ' + error.message);
  }
};

export const getEventRegistrations = async (eventId) => {
  try {
    const registrationsRef = collection(db, 'events', eventId, 'registrations');
    const querySnapshot = await getDocs(registrationsRef);
    return querySnapshot.docs.map(doc => ({
      id: doc.id,
      ...doc.data()
    }));
  } catch (error) {
    throw new Error('Error fetching event registrations: ' + error.message);
  }
};

export const updateRegistrationStatus = async (eventId, userId, status) => {
  try {
    const registrationRef = doc(db, 'events', eventId, 'registrations', userId);
    await updateDoc(registrationRef, {
      status,
      updatedAt: serverTimestamp(),
    });
  } catch (error) {
    throw new Error('Error updating registration status: ' + error.message);
  }
};

// Feedback Operations
export const submitFeedback = async (eventId, userId, feedbackData) => {
  try {
    const feedbackRef = collection(db, 'events', eventId, 'feedback');
    await setDoc(doc(feedbackRef, userId), {
      ...feedbackData,
      userId,
      submittedAt: serverTimestamp(),
    });
  } catch (error) {
    throw new Error('Error submitting feedback: ' + error.message);
  }
};

export const getEventFeedback = async (eventId) => {
  try {
    const feedbackRef = collection(db, 'events', eventId, 'feedback');
    const querySnapshot = await getDocs(feedbackRef);
    return querySnapshot.docs.map(doc => ({
      id: doc.id,
      ...doc.data()
    }));
  } catch (error) {
    throw new Error('Error fetching event feedback: ' + error.message);
  }
}; 