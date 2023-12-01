import React from 'react';

const PopularPlaces: React.FC = () => {
  return (
    <section id="popular_nav">
      {/* Popular Places */}
      <div className="popular section-bg img-effect mt-5">
        <div className="container">
          <div className="row">
            <div className="col-xl-12">
              <div className="section-title text-center">
                <h1 className="text-white">Popular Cultures </h1>
              </div>
            </div>
          </div>
        </div>
      </div>
      {/*/ End of Popular Places*/}
      {/* Other Popular */}
      <div className="all-image">
        <div className="container">
          <div className="flex flex-wrap -mx-4">
            {/* Single Image 1 */}
            <div className="w-full sm:w-1/2 lg:w-1/3 px-4 mb-8">
              <div className="rounded overflow-hidden bg-gray-800 hover:bg-opacity-75 transition duration-300">
                <img
                  src="https://i.postimg.cc/FK4dtp7z/img-1.jpg"
                  alt=""
                  className="w-full h-72 object-cover"
                />
                <div className="px-6 py-4">
                  <h3 className="text-white text-xl font-bold">Lorem Ipsum</h3>
                </div>
              </div>
            </div>
            {/* Single Image 2 */}
            <div className="w-full sm:w-1/2 lg:w-1/3 px-4 mb-8">
              <div className="rounded overflow-hidden bg-gray-800 hover:bg-opacity-75 transition duration-300">
                <img
                  src="https://i.postimg.cc/gkx63rCz/img-2.jpg"
                  alt=""
                  className="w-full h-72 object-cover"
                />
                <div className="px-6 py-4">
                  <h3 className="text-white text-xl font-bold">Lorem Ipsum</h3>
                </div>
              </div>
            </div>
            {/* Single Image 3 */}
            <div className="w-full sm:w-1/2 lg:w-1/3 px-4 mb-8">
              <div className="rounded overflow-hidden bg-gray-800 hover:bg-opacity-75 transition duration-300">
                <img
                  src="https://i.postimg.cc/Hnd1dJbQ/img-3.jpg"
                  alt=""
                  className="w-full h-72 object-cover"
                />
                <div className="px-6 py-4">
                  <h3 className="text-white text-xl font-bold">Lorem Ipsum</h3>
                </div>
              </div>
            </div>
            {/* Single Image 4 */}
            <div className="w-full sm:w-1/2 lg:w-1/3 px-4 mb-8">
              <div className="rounded overflow-hidden bg-gray-800 hover:bg-opacity-75 transition duration-300">
                <img
                  src="https://i.postimg.cc/2SwbDvvx/img-4.jpg"
                  alt=""
                  className="w-full h-72 object-cover"
                />
                <div className="px-6 py-4">
                  <h3 className="text-white text-xl font-bold">Lorem Ipsum</h3>
                </div>
              </div>
            </div>
            {/* Single Image 5 */}
            <div className="w-full sm:w-1/2 lg:w-1/3 px-4 mb-8">
              <div className="rounded overflow-hidden bg-gray-800 hover:bg-opacity-75 transition duration-300">
                <img
                  src="https://i.postimg.cc/TwmTRMFX/img-5.jpg"
                  alt=""
                  className="w-full h-72 object-cover"
                />
                <div className="px-6 py-4">
                  <h3 className="text-white text-xl font-bold">Lorem Ipsum</h3>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default PopularPlaces;
