// src/auth/firebase.js
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";


// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyC9PeGYw_ka3l207qm3fCoWGGoJqAk4RNQ",
  authDomain: "smartclassroomplatform.firebaseapp.com",
  projectId: "smartclassroomplatform",
  storageBucket: "smartclassroomplatform.firebasestorage.app",
  messagingSenderId: "1069513538420",
  appId: "1:1069513538420:web:b2bbffe1d07f3aadac0a21",
  measurementId: "G-03QQ8CHPGL"
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app);
