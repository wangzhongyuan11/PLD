// components/privacy-guard/privacy-guard.js
var app = getApp();
Component({
  data: { need: false },

  lifetimes: {
    attached: function () {
      var self = this;
      this._unregister = app.registerPrivacy(function (need) {
        self.setData({ need: need });
      });
    },
    detached: function () {
      if (this._unregister) this._unregister();
    },
  },

  methods: {
    openContract: function () { wx.openPrivacyContract({ fail: function (err) { console.error(err); } }); },
    onAgree: function () {
      app.agreePrivacy();
    },
    onRefuse: function () {
      app.refusePrivacy();
    },
  },
});
