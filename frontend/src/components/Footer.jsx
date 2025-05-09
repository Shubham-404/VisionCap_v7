const Footer = () => {
  return (
    <footer className="bg-gray-900 border-t border-gray-200 py-8 text-center text-sm text-gray-500">
      <div className="max-w-5xl mx-auto px-4">
        <p className="mb-2">© {new Date().getFullYear()} VisionCapture. All rights reserved.</p>
        <p className="mb-1">Crafted with precision for video-based behavioral analytics.</p>
        <p>Built at SybmIOT, VVCE Mysuru.</p>
      </div>
    </footer>
  );
};

export default Footer;