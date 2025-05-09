import { Link } from 'react-router-dom';

function NotFound() {
    return (
        <section className="min-h-[80vh] h-full flex items-center justify-center bg-gradient-to-br from-white to-blue-500 bg-no-repeat bg-bottom bg-cover">
            <div className="bg-white/90 backdrop-blur-xs w-full max-w-md p-8 rounded-xl shadow-lg text-center">
                <h2 className="text-3xl font-bold text-blue-700 mb-4">404 - Page Not Found</h2>
                <p className="text-gray-600 mb-6">The page you're looking for doesn't exist.</p>
                <Link
                    to="/"
                    className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition"
                >
                    Back to Home
                </Link>
            </div>
        </section>
    );
}

export default NotFound;