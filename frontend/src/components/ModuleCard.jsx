import PropTypes from 'prop-types';

const ModuleCard = ({ title, description, imgUrl, isSelected, onToggle }) => {
  return (
    <div className="bg-white p-6 rounded-xl shadow-lg hover:shadow-xl transition duration-300">
      <img
        src={imgUrl}
        alt={title}
        className="w-full h-40 object-cover rounded-lg mb-4"
      />
      <h3 className="text-xl font-semibold text-gray-800 mb-2">{title}</h3>
      <p className="text-sm text-gray-600 mb-4">{description}</p>
      {onToggle && (
        <label className="flex items-center space-x-2">
          <input
            type="checkbox"
            checked={isSelected}
            onChange={onToggle}
            className="h-5 w-5 text-blue-600 rounded focus:ring-blue-600"
          />
          <span className="text-sm font-medium text-gray-700">
            {isSelected ? 'Selected' : 'Select'}
          </span>
        </label>
      )}
    </div>
  );
};

ModuleCard.propTypes = {
  title: PropTypes.string.isRequired,
  description: PropTypes.string.isRequired,
  imgUrl: PropTypes.string.isRequired,
  isSelected: PropTypes.bool.isRequired,
  onToggle: PropTypes.func,
};

export default ModuleCard;