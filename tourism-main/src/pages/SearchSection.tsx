import React from 'react';

const SearchSection: React.FC = () => {
  return (
    <section>
      <div className="section-search">
        <div className="container">
          <div className="row">
            <div className="col-md-12 tab-search">
              <div className="nav nav-pills flex-column" id="v-pills-tab" role="tablist" aria-orientation="vertical">
                <a className="nav-link p-3 active" id="v-pills-home-tab" data-toggle="pill" href="#v-pills-home" role="tab" aria-controls="v-pills-home" aria-selected="true">Travel</a>
                <a className="nav-link p-3" id="v-pills-profile-tab" data-toggle="pill" href="#v-pills-profile" role="tab" aria-controls="v-pills-profile" aria-selected="false">Hotels</a>
                <a className="nav-link p-3" id="v-pills-messages-tab" data-toggle="pill" href="#v-pills-messages" role="tab" aria-controls="v-pills-messages" aria-selected="false">Car Rents</a>
                <a className="nav-link p-3" id="v-pills-settings-tab" data-toggle="pill" href="#v-pills-settings" role="tab" aria-controls="v-pills-settings" aria-selected="false">Cruises</a>
                <a className="nav-link p-3" id="v-pills-settings-tab" data-toggle="pill" href="#v-pills-settings" role="tab" aria-controls="v-pills-settings" aria-selected="false">Destination</a>
              </div>

              <div className="tab-content py-5" id="v-pills-tabContent">
                <div className="tab-pane fade show active" id="v-pills-home" role="tabpanel" aria-labelledby="v-pills-home-tab">
                  <div className="block-17">
                    <form className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="textfield-search">
                        <input type="text" className="form-input" placeholder="Search Location" />
                      </div>
                      <div className="check-in">
                        <input type="text" id="checkin_date" className="form-input" placeholder="Check-in date" />
                      </div>
                      <div className="check-out">
                        <input type="text" id="checkout_date" className="form-input" placeholder="Check-out date" />
                      </div>
                      <div className="select-wrap">
                        <select name="" id="" className="form-select">
                          <option value="">Guest</option>
                          <option value="">1</option>
                          <option value="">2</option>
                          <option value="">3+</option>
                        </select>
                      </div>
                      <button type="submit" className="search-submit btn btn-primary">Find Flights</button>
                    </form>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default SearchSection;
