import React from 'react';

const StatsPanel = ({ title, value, icon: Icon, color = 'blue' }) => {
  const getColorClasses = (color) => {
    switch (color) {
      case 'red':
        return {
          bg: 'bg-red-50',
          icon: 'text-red-600',
          text: 'text-red-600'
        };
      case 'green':
        return {
          bg: 'bg-green-50',
          icon: 'text-green-600',
          text: 'text-green-600'
        };
      case 'yellow':
        return {
          bg: 'bg-yellow-50',
          icon: 'text-yellow-600',
          text: 'text-yellow-600'
        };
      case 'blue':
      default:
        return {
          bg: 'bg-blue-50',
          icon: 'text-blue-600',
          text: 'text-blue-600'
        };
    }
  };

  const colorClasses = getColorClasses(color);

  return (
    <div className="card">
      <div className="card-body">
        <div className="flex items-center">
          <div className={`${colorClasses.bg} p-3 rounded-lg`}>
            <Icon className={`h-6 w-6 ${colorClasses.icon}`} />
          </div>
          <div className="ml-4">
            <p className="text-sm font-medium text-gray-600">{title}</p>
            <p className={`text-2xl font-bold ${colorClasses.text}`}>
              {value}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StatsPanel;