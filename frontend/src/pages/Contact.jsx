import React, { useState } from 'react';

const Contact = () => {
  const [form, setForm] = useState({ name: '', email: '', message: '' });
  const [status, setStatus] = useState('');

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setStatus('Submitting...');
    // Simulate API call
    setTimeout(() => {
      setStatus('Message sent successfully!');
      setForm({ name: '', email: '', message: '' });
    }, 1000);
  };

  return (
    <section className="min-h-screen bg-gradient-to-b from-slate-200 to-slate-100 text-gray-800 px-6">
      <div className="max-w-3xl mx-auto text-center pt-24 pb-16 animate-fadeIn">
        <form
          onSubmit={handleSubmit}
          className="bg-white p-8 rounded-xl shadow-lg max-w-xl mx-auto text-left"
        >
          <h1 className="text-5xl font-bold mb-6">Get in Touch</h1>
          <p className="text-lg text-gray-600 mb-8">
            Whether you're a partner, client, or curious innovator, we'd love to hear from you.
          </p>
          {status && <div className="text-green-700 mb-4 text-sm">{status}</div>}

          <label className="block text-sm font-semibold mb-1">Name</label>
          <input
            name="name"
            value={form.name}
            onChange={handleChange}
            className="w-full mb-4 px-3 py-2 border border-gray-400 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
            required
          />

          <label className="block text-sm font-semibold mb-1">Email</label>
          <input
            type="email"
            name="email"
            value={form.email}
            onChange={handleChange}
            className="w-full mb-4 px-3 py-2 border border-gray-400 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
            required
          />

          <label className="block text-sm font-semibold mb-1">Message</label>
          <textarea
            name="message"
            value={form.message}
            onChange={handleChange}
            rows="4"
            className="w-full mb-6 px-3 py-2 border border-gray-400 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
            required
          />

          <button
            type="submit"
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-lg shadow transition hover:scale-105"
          >
            Send Message
          </button>
        </form>
      </div>
    </section>
  );
};

export default Contact;
